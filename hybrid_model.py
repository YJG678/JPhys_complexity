"""
Hybrid agent-particle chemotaxis simulator (2D).

Implements the microscopic model of the manuscript:

  dot X_i = v0 * omega_i + mu_p * (1/N) sum_{j != i} K(X_i - X_j)     [+ chi grad rho]
  dot M_i = (1/tau_m) (rho(X_i) - M_i)
  omega_i tumbles at rate lambda(rho(X_i) - M_i)

on the periodic torus [0,L)^2. Three force back-ends are provided and are
numerically interchangeable:

  * direct      : O(N^2) all-pairs summation (reference / ground truth)
  * cell-list   : O(N) search for the compactly supported short-range kernel
  * barnes_hut  : O(N log N) monopole tree approximation for the long-range kernel

Weight convention (per the corrected Barnes-Hut analysis): every particle
carries unit weight, so the cell aggregate M_Q is the particle count and c_Q
is the geometric centroid. The overall mean-field factor 1/N multiplies the
assembled interaction field.
"""

import numpy as np


# ----------------------------------------------------------------------
# Periodic geometry helpers (minimum-image convention on [0,L)^d)
# ----------------------------------------------------------------------
def min_image(dx, L):
    """Wrap displacement components to [-L/2, L/2)."""
    return dx - L * np.round(dx / L)


def periodic_dist_to_point(X, c, L):
    d = min_image(X - c[None, :], L)
    return np.sqrt(np.sum(d * d, axis=1))


# ----------------------------------------------------------------------
# Nutrient field  rho(x) = 1 / (1 + |x - C|_per)
# ----------------------------------------------------------------------
def nutrient(X, C, L):
    return 1.0 / (1.0 + periodic_dist_to_point(X, C, L))


def nutrient_grad(X, C, L):
    """grad of 1/(1+r):  -1/(1+r)^2 * (x-C)/r ."""
    d = min_image(X - C[None, :], L)
    r = np.sqrt(np.sum(d * d, axis=1)) + 1e-12
    coeff = -1.0 / (1.0 + r) ** 2 / r
    return coeff[:, None] * d


# ----------------------------------------------------------------------
# Tumbling rate  lambda(q) = lambda0 - lambda1 tanh(kappa q),  q = rho - M
# ----------------------------------------------------------------------
def tumble_rate(q, lam0, lam1, kappa):
    return lam0 - lam1 * np.tanh(kappa * q)


# ----------------------------------------------------------------------
# Interaction kernels (return the force / velocity contribution vector)
# Short range: soft compactly-supported repulsion, support r_c.
#   K_s(z) = eps_s * (1 - |z|/r_c) * z/|z|      (points away from neighbour)
# Long range: bounded, smooth attraction with regularisation eps.
#   K_l(z) = -eps_l * z / (|z|^2 + reg^2)       (points toward neighbour)
# Here z = X_i - X_j, so the returned vector is the contribution to dot X_i.
# ----------------------------------------------------------------------
def short_range_pair(z, dist, eps_s, r_c):
    """z: (m,2) displacements X_i - X_j (min-image), dist: (m,) norms."""
    mag = eps_s * (1.0 - dist / r_c)
    out = (mag / (dist + 1e-12))[:, None] * z
    return out


def long_range_kernel(z, eps_l, reg):
    """Attractive bounded kernel evaluated on displacements z = X_i - X_j."""
    d2 = np.sum(z * z, axis=-1)
    coeff = -eps_l / (d2 + reg * reg)
    return coeff[..., None] * z


# ======================================================================
# Direct O(N^2) short-range force (reference)
# ======================================================================
def short_range_direct(X, L, eps_s, r_c):
    N = X.shape[0]
    F = np.zeros_like(X)
    for i in range(N):
        z = min_image(X[i][None, :] - X, L)
        dist = np.sqrt(np.sum(z * z, axis=1))
        mask = (dist <= r_c) & (dist > 0)
        if np.any(mask):
            F[i] = short_range_pair(z[mask], dist[mask], eps_s, r_c).sum(axis=0)
    return F / N


# ======================================================================
# Cell-list O(N) short-range force
# ======================================================================
def short_range_celllist(X, L, eps_s, r_c):
    N = X.shape[0]
    ncell = max(1, int(np.floor(L / r_c)))          # cell width h = L/ncell >= r_c
    h = L / ncell
    cix = np.floor(X[:, 0] / h).astype(int) % ncell
    ciy = np.floor(X[:, 1] / h).astype(int) % ncell
    cell_of = ciy * ncell + cix
    # bucket particles by cell
    order = np.argsort(cell_of, kind="stable")
    cell_sorted = cell_of[order]
    starts = np.searchsorted(cell_sorted, np.arange(ncell * ncell))
    ends = np.searchsorted(cell_sorted, np.arange(ncell * ncell) + 1)
    F = np.zeros_like(X)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
            (0, 1), (1, -1), (1, 0), (1, 1)]
    occupied = np.unique(cell_of)          # only visit non-empty cells
    for c in occupied:
        cx = c % ncell
        cy = c // ncell
        idx_i = order[starts[c]:ends[c]]
        if idx_i.size == 0:
            continue
        neigh = []
        for (ox, oy) in offs:
            nx = (cx + ox) % ncell
            ny = (cy + oy) % ncell
            cc = ny * ncell + nx
            neigh.append(order[starts[cc]:ends[cc]])
        idx_j = np.concatenate(neigh)
        Xi = X[idx_i]          # (ni,2)
        Xj = X[idx_j]          # (nj,2)
        z = min_image(Xi[:, None, :] - Xj[None, :, :], L)   # (ni,nj,2)
        dist = np.sqrt(np.sum(z * z, axis=2))
        mask = (dist <= r_c) & (dist > 0)
        mag = np.where(mask, eps_s * (1.0 - dist / r_c) / (dist + 1e-12), 0.0)
        contrib = (mag[:, :, None] * z).sum(axis=1)
        F[idx_i] += contrib
    return F / N


# ======================================================================
# Quadtree Barnes-Hut O(N log N) long-range force
# (unit weights; monopole/centroid approximation; opening ratio s/d < theta)
# Built on the *unwrapped* box [0,L)^2. Long-range attraction is smooth and
# bounded, so the far field is well behaved.
# ======================================================================
class QuadNode:
    __slots__ = ("x0", "y0", "s", "count", "cx", "cy", "children", "px", "py")

    def __init__(self, x0, y0, s):
        self.x0 = x0
        self.y0 = y0
        self.s = s
        self.count = 0
        self.cx = 0.0     # accumulated sum of x (centroid numerator)
        self.cy = 0.0
        self.children = None
        self.px = None    # stored single-point coords if leaf with one body
        self.py = None


def _insert(node, x, y, depth, max_depth):
    node.count += 1
    node.cx += x
    node.cy += y
    if node.count == 1:
        node.px, node.py = x, y
        return
    if depth >= max_depth:
        return
    if node.children is None:
        node.children = [None, None, None, None]
        # reinsert the previously stored single body
        if node.px is not None:
            _insert_child(node, node.px, node.py, depth, max_depth)
            node.px = node.py = None
    _insert_child(node, x, y, depth, max_depth)


def _insert_child(node, x, y, depth, max_depth):
    half = node.s * 0.5
    qx = 0 if x < node.x0 + half else 1
    qy = 0 if y < node.y0 + half else 1
    q = qy * 2 + qx
    if node.children[q] is None:
        node.children[q] = QuadNode(node.x0 + qx * half, node.y0 + qy * half, half)
    _insert(node.children[q], x, y, depth + 1, max_depth)


def build_quadtree(X, L, max_depth=20):
    root = QuadNode(0.0, 0.0, L)
    for k in range(X.shape[0]):
        _insert(root, X[k, 0], X[k, 1], 0, max_depth)
    return root


def _bh_force_on(node, xi, yi, theta, eps_l, reg, L):
    """Accumulate long-range force on target (xi,yi) from node, min-image."""
    if node is None or node.count == 0:
        return 0.0, 0.0
    ccx = node.cx / node.count
    ccy = node.cy / node.count
    dx = min_image(np.array(xi - ccx), L)
    dy = min_image(np.array(yi - ccy), L)
    d = np.sqrt(dx * dx + dy * dy)
    if node.children is None or (node.s < theta * d):
        if d < 1e-9:
            return 0.0, 0.0
        coeff = -eps_l * node.count / (d * d + reg * reg)
        return coeff * dx, coeff * dy
    fx = fy = 0.0
    for c in node.children:
        cx, cy = _bh_force_on(c, xi, yi, theta, eps_l, reg, L)
        fx += cx
        fy += cy
    return fx, fy


def long_range_barnes_hut(X, L, eps_l, reg, theta):
    N = X.shape[0]
    root = build_quadtree(X, L)
    F = np.zeros_like(X)
    for i in range(N):
        fx, fy = _bh_force_on(root, X[i, 0], X[i, 1], theta, eps_l, reg, L)
        F[i, 0] = fx
        F[i, 1] = fy
    return F / N


def long_range_direct(X, L, eps_l, reg):
    N = X.shape[0]
    F = np.zeros_like(X)
    for i in range(N):
        z = min_image(X[i][None, :] - X, L)
        z[i] = 0.0
        F[i] = long_range_kernel(z, eps_l, reg).sum(axis=0)
    return F / N


def long_range_direct_vec(X, L, eps_l, reg):
    """Vectorised O(N^2) long-range force; fast for moderate N (dynamics use)."""
    N = X.shape[0]
    z = min_image(X[:, None, :] - X[None, :, :], L)     # (N,N,2)
    d2 = np.sum(z * z, axis=2)
    coeff = -eps_l / (d2 + reg * reg)
    np.fill_diagonal(coeff, 0.0)
    F = (coeff[:, :, None] * z).sum(axis=1)
    return F / N


def short_range_direct_vec(X, L, eps_s, r_c):
    """Vectorised O(N^2) short-range force (dynamics use for small N)."""
    N = X.shape[0]
    z = min_image(X[:, None, :] - X[None, :, :], L)
    dist = np.sqrt(np.sum(z * z, axis=2))
    mag = np.where((dist <= r_c) & (dist > 0),
                   eps_s * (1.0 - dist / r_c) / (dist + 1e-12), 0.0)
    F = (mag[:, :, None] * z).sum(axis=1)
    return F / N


# ======================================================================
# Simulator
# ======================================================================
class HybridSim:
    def __init__(self, N, L=1.0, d=2, seed=0,
                 v0=0.05, dt=0.05, tau_m=0.5,
                 lam0=2.0, lam1=1.8, kappa=8.0,
                 mu_p=1.0, chi=0.0,
                 eps_s=0.0, r_c=0.06,
                 eps_l=0.0, reg=0.02,
                 theta=0.5, C=None,
                 short_backend="celllist", long_backend="direct"):
        self.N, self.L, self.dt = N, L, dt
        self.v0, self.tau_m = v0, tau_m
        self.lam0, self.lam1, self.kappa = lam0, lam1, kappa
        self.mu_p, self.chi = mu_p, chi
        self.eps_s, self.r_c = eps_s, r_c
        self.eps_l, self.reg, self.theta = eps_l, reg, theta
        self.short_backend = short_backend
        self.long_backend = long_backend
        self.rng = np.random.default_rng(seed)
        self.C = np.array([0.5, 0.5]) if C is None else np.asarray(C)

        self.X = self.rng.random((N, 2)) * L
        ang = self.rng.random(N) * 2 * np.pi
        self.omega = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.M = nutrient(self.X, self.C, self.L)   # remember starting value

    def _short_force(self):
        if self.eps_s == 0.0:
            return np.zeros_like(self.X)
        if self.short_backend == "direct":
            return short_range_direct(self.X, self.L, self.eps_s, self.r_c)
        if self.short_backend == "direct_vec":
            return short_range_direct_vec(self.X, self.L, self.eps_s, self.r_c)
        return short_range_celllist(self.X, self.L, self.eps_s, self.r_c)

    def _long_force(self):
        if self.eps_l == 0.0:
            return np.zeros_like(self.X)
        if self.long_backend == "barnes_hut":
            return long_range_barnes_hut(self.X, self.L, self.eps_l,
                                         self.reg, self.theta)
        if self.long_backend == "direct_vec":
            return long_range_direct_vec(self.X, self.L, self.eps_l, self.reg)
        return long_range_direct(self.X, self.L, self.eps_l, self.reg)

    def step(self):
        L, dt = self.L, self.dt
        rho_i = nutrient(self.X, self.C, L)
        q = rho_i - self.M
        # behavioural update
        lam = tumble_rate(q, self.lam0, self.lam1, self.kappa)
        p = 1.0 - np.exp(-lam * dt)
        tumble = self.rng.random(self.N) < p
        if np.any(tumble):
            na = self.rng.random(tumble.sum()) * 2 * np.pi
            self.omega[tumble] = np.stack([np.cos(na), np.sin(na)], axis=1)
        # forces
        F = self.mu_p * (self._short_force() + self._long_force())
        drift = self.chi * nutrient_grad(self.X, self.C, L) if self.chi else 0.0
        # transport
        self.X = (self.X + dt * (self.v0 * self.omega + F + drift)) % L
        # memory (exact relaxation)
        a = np.exp(-dt / self.tau_m)
        self.M = a * self.M + (1 - a) * rho_i

    def run(self, nsteps, record_every=0, metrics=None):
        hist = {"t": []}
        if metrics:
            for m in metrics:
                hist[m] = []
        for n in range(nsteps):
            self.step()
            if record_every and (n % record_every == 0 or n == nsteps - 1):
                hist["t"].append(n * self.dt)
                if metrics:
                    for m in metrics:
                        hist[m].append(metrics[m](self))
        return hist


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def R_c(sim):
    return periodic_dist_to_point(sim.X, sim.C, sim.L).mean()


def aggregation_index(sim, r_a=0.05):
    """Mean neighbour count within r_a, normalised by random expectation."""
    N, L = sim.N, sim.L
    exp_rand = (N - 1) * np.pi * r_a ** 2 / (L * L)
    # cell-list count
    ncell = max(1, int(np.floor(L / r_a)))
    h = L / ncell
    cix = np.floor(sim.X[:, 0] / h).astype(int) % ncell
    ciy = np.floor(sim.X[:, 1] / h).astype(int) % ncell
    cell_of = ciy * ncell + cix
    order = np.argsort(cell_of, kind="stable")
    cs = cell_of[order]
    starts = np.searchsorted(cs, np.arange(ncell * ncell))
    ends = np.searchsorted(cs, np.arange(ncell * ncell) + 1)
    total = 0
    offs = [(ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1)]
    for cy in range(ncell):
        for cx in range(ncell):
            c = cy * ncell + cx
            idx_i = order[starts[c]:ends[c]]
            if idx_i.size == 0:
                continue
            neigh = []
            for (ox, oy) in offs:
                cc = ((cy + oy) % ncell) * ncell + ((cx + ox) % ncell)
                neigh.append(order[starts[cc]:ends[cc]])
            idx_j = np.concatenate(neigh)
            z = min_image(sim.X[idx_i][:, None, :] - sim.X[idx_j][None, :, :], L)
            dist = np.sqrt(np.sum(z * z, axis=2))
            total += int(np.sum((dist <= r_a) & (dist > 0)))
    mean_neigh = total / N
    return mean_neigh / max(exp_rand, 1e-12)


def kl_to_nutrient(sim, nbins=32):
    """D_KL( n(t,.) || rho(.) ) on a coarse grid, both normalised to prob."""
    L = sim.L
    H, _, _ = np.histogram2d(sim.X[:, 0], sim.X[:, 1],
                             bins=nbins, range=[[0, L], [0, L]])
    n = H / H.sum()
    # nutrient on cell centres
    ce = (np.arange(nbins) + 0.5) * L / nbins
    gx, gy = np.meshgrid(ce, ce, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    r = nutrient(pts, sim.C, L).reshape(nbins, nbins)
    r = r / r.sum()
    eps = 1e-12
    mask = n > 0
    return float(np.sum(n[mask] * np.log((n[mask] + eps) / (r[mask] + eps))))
