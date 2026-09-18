"""d-dimensional hybrid chemotaxis simulator (d = 2 or 3) with vectorised direct
forces for the physics comparison, plus an octree Barnes-Hut long-range force
(3D, unit weights, centroid monopole) for the computational comparison.

The octree generalises the provided 3D Barnes-Hut implementation: eight children
per node, opening ratio s/d < theta, aggregate = particle count at the geometric
centroid (unit weights, mean-field 1/N normalisation)."""
import numpy as np


def min_image(dx, L):
    return dx - L * np.round(dx / L)


def periodic_dist_to_point(X, c, L):
    d = min_image(X - c[None, :], L)
    return np.sqrt(np.sum(d * d, axis=1))


def nutrient(X, C, L):
    return 1.0 / (1.0 + periodic_dist_to_point(X, C, L))


def tumble_rate(q, lam0, lam1, kappa):
    return lam0 - lam1 * np.tanh(kappa * q)


def sample_directions(rng, n, d):
    if d == 2:
        a = rng.random(n) * 2 * np.pi
        return np.stack([np.cos(a), np.sin(a)], axis=1)
    z = rng.uniform(-1, 1, n)
    phi = rng.random(n) * 2 * np.pi
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def short_range_vec(X, L, eps_s, r_c):
    z = min_image(X[:, None, :] - X[None, :, :], L)
    dist = np.sqrt(np.sum(z * z, axis=2))
    mag = np.where((dist <= r_c) & (dist > 0),
                   eps_s * (1.0 - dist / r_c) / (dist + 1e-12), 0.0)
    return (mag[:, :, None] * z).sum(axis=1) / X.shape[0]


class HybridSimND:
    def __init__(self, N, d=2, L=1.0, seed=0, v0=0.12, dt=0.05, tau_m=0.25,
                 lam0=5.0, lam1=4.5, kappa=20.0, mu_p=1.0,
                 eps_s=0.0, r_c=0.06, C=None):
        self.N, self.d, self.L, self.dt = N, d, L, dt
        self.v0, self.tau_m = v0, tau_m
        self.lam0, self.lam1, self.kappa = lam0, lam1, kappa
        self.mu_p, self.eps_s, self.r_c = mu_p, eps_s, r_c
        self.rng = np.random.default_rng(seed)
        self.C = (np.full(d, 0.5) if C is None else np.asarray(C))
        self.X = self.rng.random((N, d)) * L
        self.omega = sample_directions(self.rng, N, d)
        self.M = nutrient(self.X, self.C, L)

    def step(self):
        L, dt = self.L, self.dt
        rho_i = nutrient(self.X, self.C, L)
        q = rho_i - self.M
        lam = tumble_rate(q, self.lam0, self.lam1, self.kappa)
        p = 1.0 - np.exp(-lam * dt)
        tb = self.rng.random(self.N) < p
        if np.any(tb):
            self.omega[tb] = sample_directions(self.rng, int(tb.sum()), self.d)
        F = np.zeros_like(self.X)
        if self.eps_s:
            F = self.mu_p * short_range_vec(self.X, L, self.eps_s, self.r_c)
        self.X = (self.X + dt * (self.v0 * self.omega + F)) % L
        a = np.exp(-dt / self.tau_m)
        self.M = a * self.M + (1 - a) * rho_i

    def run(self, nsteps, record_every=0):
        hist = {"t": [], "Rc": []}
        for n in range(nsteps):
            self.step()
            if record_every and (n % record_every == 0 or n == nsteps - 1):
                hist["t"].append(n * self.dt)
                hist["Rc"].append(periodic_dist_to_point(self.X, self.C, L=self.L).mean())
        return hist


def R_c(sim):
    return periodic_dist_to_point(sim.X, sim.C, sim.L).mean()


def clustering_index(sim, r_a=0.06):
    """Neighbour count within r_a normalised by random expectation (d-dependent)."""
    N, L, d = sim.N, sim.L, sim.d
    if d == 2:
        vol = np.pi * r_a ** 2
    else:
        vol = 4.0 / 3.0 * np.pi * r_a ** 3
    exp_rand = (N - 1) * vol / L ** d
    z = min_image(sim.X[:, None, :] - sim.X[None, :, :], L)
    dist = np.sqrt(np.sum(z * z, axis=2))
    cnt = np.sum((dist <= r_a) & (dist > 0)) / N
    return cnt / max(exp_rand, 1e-12)


# ----------------------------------------------------------------------
# Octree Barnes-Hut (3D), unit weights, centroid monopole
# ----------------------------------------------------------------------
class OctNode:
    __slots__ = ("o", "s", "count", "csum", "children", "p")

    def __init__(self, o, s):
        self.o = o            # origin (corner) length-3 array
        self.s = s            # side length
        self.count = 0
        self.csum = np.zeros(3)
        self.children = None
        self.p = None


def _oct_insert(node, x, depth, max_depth):
    node.count += 1
    node.csum += x
    if node.count == 1:
        node.p = x
        return
    if depth >= max_depth:
        return
    if node.children is None:
        node.children = [None] * 8
        if node.p is not None:
            _oct_child(node, node.p, depth, max_depth)
            node.p = None
    _oct_child(node, x, depth, max_depth)


def _oct_child(node, x, depth, max_depth):
    half = node.s * 0.5
    bx = 0 if x[0] < node.o[0] + half else 1
    by = 0 if x[1] < node.o[1] + half else 1
    bz = 0 if x[2] < node.o[2] + half else 1
    q = bz * 4 + by * 2 + bx
    if node.children[q] is None:
        origin = node.o + np.array([bx, by, bz]) * half
        node.children[q] = OctNode(origin, half)
    _oct_insert(node.children[q], x, depth + 1, max_depth)


def build_octree(X, L, max_depth=20):
    root = OctNode(np.zeros(3), L)
    for k in range(X.shape[0]):
        _oct_insert(root, X[k], 0, max_depth)
    return root


def _oct_force(node, xi, theta, eps_l, reg, L):
    if node is None or node.count == 0:
        return np.zeros(3)
    c = node.csum / node.count
    dz = min_image(xi - c, L)
    d = np.sqrt(dz @ dz)
    if node.children is None or node.s < theta * d:
        if d < 1e-9:
            return np.zeros(3)
        return (-eps_l * node.count / (d * d + reg * reg)) * dz
    f = np.zeros(3)
    for ch in node.children:
        f += _oct_force(ch, xi, theta, eps_l, reg, L)
    return f


def long_range_octree(X, L, eps_l, reg, theta):
    root = build_octree(X, L)
    F = np.zeros_like(X)
    for i in range(X.shape[0]):
        F[i] = _oct_force(root, X[i], theta, eps_l, reg, L)
    return F / X.shape[0]


def long_range_direct3d(X, L, eps_l, reg):
    z = min_image(X[:, None, :] - X[None, :, :], L)
    d2 = np.sum(z * z, axis=2)
    coeff = -eps_l / (d2 + reg * reg)
    np.fill_diagonal(coeff, 0.0)
    return (coeff[:, :, None] * z).sum(axis=1) / X.shape[0]
