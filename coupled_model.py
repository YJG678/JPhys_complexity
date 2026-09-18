"""
Coupled particle-field solver -- the model Eq. (coupled-system) / Algorithm 1
of the manuscript actually states.

WHY THIS FILE EXISTS
--------------------
The submitted `hybrid_model.HybridSim` evaluates the nutrient as a FIXED analytic
landscape rho(x) = 1/(1 + |x-C|). It carries no field state, no uptake deposition
and no field solve, so the reaction-diffusion equation

    d_t rho = D_rho Lap rho - alpha rho - beta rho U^N + S,
    U^N(x)  = (1/N) sum_j phi_eps(x - X_j),        int U^N dx = 1

is not implemented anywhere in the submitted code, and neither are steps 7-8 of
Algorithm 1. This module implements them.

WHAT ELSE CHANGED
-----------------
1. SCREENED long-range kernel.  The submitted kernel K = -eps_l z/(|z|^2+reg^2)
   decays as 1/|z| and was evaluated under the minimum-image convention with no
   image sum and no neutralising background. For that kernel the periodic sum is
   only conditionally convergent and minimum image is not a controlled
   approximation (measured: off by a factor ~9 against a 12-shell lattice sum).
   The kernel is now Yukawa-screened,

       K_l(z) = -eps_l * z * exp(-kappa |z|) / (|z|^2 + reg^2),

   with kappa = 1/xi and xi = sqrt(D_rho/(alpha + beta n0)) the screening length
   the nutrient uptake generates anyway. The periodic image sum then converges
   geometrically and minimum image is controlled to O(exp(-kappa L/2)).
   kappa = 0 recovers the submitted kernel exactly (for backward comparison).

2. Short-range kernel regularised at the origin so that K_s is in W^{1,inf},
   as Assumption 1 of the manuscript requires:
       K_s(z) = eps_s (1 - |z|/r_c)_+ * z / sqrt(|z|^2 + reg_s^2)
   Setting reg_s = 0 recovers the submitted form (which is discontinuous at 0).

Discretisation (matches the manuscript):
  * particle -> grid : CIC ("cloud in cell") deposition, phi_eps = CIC cloud
  * grid -> particle : CIC interpolation (bilinear), 2nd order
  * field            : semi-implicit, diffusion+decay implicit, particle
                       reaction explicit, solved spectrally on the periodic
                       5-point Laplacian (same operator as run_grid_refine.py)
  * gradient         : centred 2nd-order differences (consistent with Lap_h)
"""

import numpy as np

# ----------------------------------------------------------------- geometry
def min_image(dx, L):
    return dx - L * np.round(dx / L)


# ------------------------------------------------------- particle <-> grid
def _cic(X, L, M):
    """CIC weights: returns 4 (ix,iy) index pairs and their weights."""
    h = L / M
    g = X / h
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    i0 %= M
    i1 = (i0 + 1) % M
    idx = [(i0[:, 0], i0[:, 1]), (i1[:, 0], i0[:, 1]),
           (i0[:, 0], i1[:, 1]), (i1[:, 0], i1[:, 1])]
    w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]),
         (1 - f[:, 0]) * f[:, 1],       f[:, 0] * f[:, 1]]
    return idx, w, h


def deposit(X, L, M):
    """U^N = (1/N) sum_j phi_eps(x - X_j) with phi_eps the CIC cloud.

    Normalised so that int U^N dx = sum_alpha U_alpha h^2 = 1 exactly.
    """
    idx, w, h = _cic(X, L, M)
    U = np.zeros((M, M))
    for (ix, iy), ww in zip(idx, w):
        np.add.at(U, (ix, iy), ww)
    return U / (X.shape[0] * h * h)


def interp(G, X, L, M):
    idx, w, _ = _cic(X, L, M)
    out = np.zeros(X.shape[0])
    for (ix, iy), ww in zip(idx, w):
        out += ww * G[ix, iy]
    return out


def interp_vec(Gx, Gy, X, L, M):
    return np.stack([interp(Gx, X, L, M), interp(Gy, X, L, M)], axis=1)


# ------------------------------------------------------------------- field
class NutrientField:
    """Semi-implicit solver for
           d_t rho = D Lap rho - alpha rho - beta rho U + S
       (I - dt D Lap_h + dt alpha) rho^{n+1} = rho^n - dt beta rho^n U^n + dt S
    diagonalised by the DFT on the periodic 5-point Laplacian."""

    def __init__(self, M, L, D_rho, alpha, beta, S, rho0):
        self.M, self.L = M, L
        self.D, self.alpha, self.beta = D_rho, alpha, beta
        self.S = np.asarray(S, float) if np.ndim(S) else np.full((M, M), float(S))
        self.rho = (np.array(rho0, float).copy() if np.ndim(rho0)
                    else np.full((M, M), float(rho0)))
        h = L / M
        lam1 = (2 * np.cos(2 * np.pi * np.arange(M) / M) - 2) / h ** 2
        self.lam = lam1[:, None] + lam1[None, :]      # eigenvalues of Lap_h
        self.h = h

    def step(self, dt, U):
        """One step, by Lie splitting:

          (i)  local uptake, solved EXACTLY:   rho* = rho^n exp(-dt beta U^n)
          (ii) diffusion + decay + source, implicit and spectral:
               (I - dt D Lap_h + dt alpha) rho^{n+1} = rho* + dt S

        Treating the uptake explicitly (as rho^n - dt beta rho^n U^n) is only
        conditionally stable: it requires dt beta max(U) < 1, and max(U) grows
        without bound as particles aggregate, so it fails in exactly the
        collapsed regime the model is meant to probe.  The exponential update is
        unconditionally stable and unconditionally positive, and the implicit
        operator is an M-matrix, so rho >= 0 is preserved by construction with
        no clamping.  The splitting is first order, matching the time
        discretisation of the rest of the scheme.
        """
        rho_star = self.rho * np.exp(-dt * self.beta * U)
        rhs = rho_star + dt * self.S
        denom = 1.0 - dt * self.D * self.lam + dt * self.alpha
        self.rho = np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))
        return self.rho

    def grad(self):
        """Centred 2nd-order gradient, consistent with the 5-point Laplacian."""
        h = self.h
        gx = (np.roll(self.rho, -1, 0) - np.roll(self.rho, 1, 0)) / (2 * h)
        gy = (np.roll(self.rho, -1, 1) - np.roll(self.rho, 1, 1)) / (2 * h)
        return gx, gy

    def screening_length(self, n0):
        """xi = sqrt(D/(alpha + beta n0)) -- Eq. (xi-here) of the manuscript."""
        denom = self.alpha + self.beta * n0
        return np.inf if denom <= 0 else np.sqrt(self.D / denom)


# ----------------------------------------------------------------- kernels
try:                                    # optional compiled fast path
    from bh_numba import short_range_celllist_numba as _celllist_fast
except Exception:                       # numba absent -> pure NumPy below
    _celllist_fast = None


def short_range_celllist(X, L, eps_s, r_c, reg_s=0.0, force_numpy=False):
    """Exact O(N) short-range force. reg_s > 0 regularises the origin so that
    K_s is in W^{1,inf} (Assumption 1); reg_s = 0 is the submitted form.

    Uses the compiled (numba) kernel when available; the pure-NumPy path is kept
    and the two agree to machine precision (test_coupled.py T9)."""
    if _celllist_fast is not None and not force_numpy:
        return _celllist_fast(np.ascontiguousarray(X, dtype=np.float64),
                              float(L), float(eps_s), float(r_c), float(reg_s))
    N = X.shape[0]
    ncell = max(1, int(np.floor(L / r_c)))
    h = L / ncell
    cix = np.floor(X[:, 0] / h).astype(int) % ncell
    ciy = np.floor(X[:, 1] / h).astype(int) % ncell
    cell_of = ciy * ncell + cix
    order = np.argsort(cell_of, kind="stable")
    cs = cell_of[order]
    starts = np.searchsorted(cs, np.arange(ncell * ncell))
    ends = np.searchsorted(cs, np.arange(ncell * ncell) + 1)
    F = np.zeros_like(X)
    offs = [(ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1)]
    for c in np.unique(cell_of):
        cx, cy = c % ncell, c // ncell
        ii = order[starts[c]:ends[c]]
        if ii.size == 0:
            continue
        jj = np.concatenate([
            order[starts[((cy + oy) % ncell) * ncell + ((cx + ox) % ncell)]:
                  ends[((cy + oy) % ncell) * ncell + ((cx + ox) % ncell)]]
            for (ox, oy) in offs])
        z = min_image(X[ii][:, None, :] - X[jj][None, :, :], L)
        d = np.sqrt((z * z).sum(2))
        denom = np.sqrt(d * d + reg_s * reg_s) if reg_s > 0 else d + 1e-12
        mag = np.where((d <= r_c) & (d > 0), eps_s * (1.0 - d / r_c) / denom, 0.0)
        F[ii] += (mag[:, :, None] * z).sum(1)
    return F / N


def long_range_screened_vec(X, L, eps_l, reg, kappa=0.0):
    """Minimum-image screened long-range force, O(N^2), vectorised.
    kappa = 0 reproduces the submitted unscreened kernel exactly."""
    N = X.shape[0]
    z = min_image(X[:, None, :] - X[None, :, :], L)
    d2 = (z * z).sum(2)
    coeff = -eps_l / (d2 + reg * reg)
    if kappa > 0:
        coeff = coeff * np.exp(-kappa * np.sqrt(d2))
    np.fill_diagonal(coeff, 0.0)
    return (coeff[:, :, None] * z).sum(1) / N


def long_range_lattice(X, L, eps_l, reg, kappa=0.0, nimg=3):
    """Reference: raw periodic lattice sum, each periodic partner counted once.
    NOTE: no minimum image anywhere -- mixing the two double-counts near pairs."""
    N = X.shape[0]
    F = np.zeros_like(X)
    d0 = X[:, None, :] - X[None, :, :]
    for a in range(-nimg, nimg + 1):
        for b in range(-nimg, nimg + 1):
            z = d0 - np.array([a * L, b * L])
            d2 = (z * z).sum(2)
            coeff = -eps_l / (d2 + reg * reg)
            if kappa > 0:
                coeff = coeff * np.exp(-kappa * np.sqrt(d2))
            if a == 0 and b == 0:
                np.fill_diagonal(coeff, 0.0)
            F += (coeff[:, :, None] * z).sum(1)
    return F / N


def tumble_rate(q, lam0, lam1, kappa_s):
    return lam0 - lam1 * np.tanh(kappa_s * q)


# --------------------------------------------------------------- simulator
class CoupledSim:
    """Algorithm 1 of the manuscript, in full.

    Step order: perceive -> turn -> force -> transport -> memory -> deposit
                -> field solve.
    """

    def __init__(self, N, L=1.0, M=128, seed=0,
                 v0=0.12, dt=0.02, tau_m=0.25,
                 lam0=5.0, lam1=4.5, kappa_s=20.0,
                 mu_p=1.0, chi=0.0,
                 eps_s=0.0, r_c=0.06, reg_s=3e-4,
                 eps_l=0.0, reg=0.03, kappa_screen=None,
                 D_rho=0.02, alpha=0.5, beta=1.0,
                 S=1.0, rho_init=None,
                 source="uniform", source_width=0.12, C=None):
        self.N, self.L, self.M, self.dt = N, L, M, dt
        self.v0, self.tau_m = v0, tau_m
        self.lam0, self.lam1, self.kappa_s = lam0, lam1, kappa_s
        self.mu_p, self.chi = mu_p, chi
        self.eps_s, self.r_c, self.reg_s = eps_s, r_c, reg_s
        self.eps_l, self.reg = eps_l, reg
        self.rng = np.random.default_rng(seed)
        self.C = np.array([0.5 * L, 0.5 * L]) if C is None else np.asarray(C, float)

        # --- source field
        xs = (np.arange(M) + 0.5) * L / M
        GX, GY = np.meshgrid(xs, xs, indexing="ij")
        if source == "uniform":
            Sfield = np.full((M, M), float(S))
        elif source == "gaussian":
            dx = min_image(GX - self.C[0], L)
            dy = min_image(GY - self.C[1], L)
            Sfield = float(S) * np.exp(-(dx ** 2 + dy ** 2) / (2 * source_width ** 2))
        else:
            raise ValueError("source must be 'uniform' or 'gaussian'")

        # --- initial field: particle-free steady state  (D Lap - alpha) rho + S = 0
        if rho_init is None:
            h = L / M
            lam1d = (2 * np.cos(2 * np.pi * np.arange(M) / M) - 2) / h ** 2
            lam = lam1d[:, None] + lam1d[None, :]
            rho0 = np.real(np.fft.ifft2(np.fft.fft2(Sfield) /
                                        (alpha - D_rho * lam)))
        else:
            rho0 = rho_init
        self.field = NutrientField(M, L, D_rho, alpha, beta, Sfield, rho0)

        # --- screening length: kappa = 1/xi from the uptake, unless overridden
        # U^N is normalised to int U = 1, so its uniform value is U0 = 1/L^2;
        # that is the n0 appearing in xi = sqrt(D/(alpha + beta n0)).
        self.U0 = 1.0 / (L * L)
        xi = self.field.screening_length(self.U0)
        self.xi = xi
        if kappa_screen is None:
            self.kappa_screen = 0.0 if (not np.isfinite(xi) or xi <= 0) else 1.0 / xi
        else:
            self.kappa_screen = float(kappa_screen)

        # --- particles
        self.X = self.rng.random((N, 2)) * L
        a = self.rng.random(N) * 2 * np.pi
        self.omega = np.stack([np.cos(a), np.sin(a)], axis=1)
        self.Mem = interp(self.field.rho, self.X, L, M)   # M_i(0) = rho(X_i(0))
        self.t = 0.0

    # ------------------------------------------------------------ one step
    def step(self):
        L, M, dt = self.L, self.M, self.dt
        # (2) perceive
        rho_i = interp(self.field.rho, self.X, L, M)
        q = rho_i - self.Mem
        # (3) turn
        p = 1.0 - np.exp(-tumble_rate(q, self.lam0, self.lam1, self.kappa_s) * dt)
        tb = self.rng.random(self.N) < p
        if np.any(tb):
            na = self.rng.random(int(tb.sum())) * 2 * np.pi
            self.omega[tb] = np.stack([np.cos(na), np.sin(na)], axis=1)
        # (4) forces
        F = np.zeros_like(self.X)
        if self.eps_s:
            F += short_range_celllist(self.X, L, self.eps_s, self.r_c, self.reg_s)
        if self.eps_l:
            F += long_range_screened_vec(self.X, L, self.eps_l, self.reg,
                                         self.kappa_screen)
        F *= self.mu_p
        # (5) transport
        drift = 0.0
        if self.chi:
            gx, gy = self.field.grad()
            drift = self.chi * interp_vec(gx, gy, self.X, L, M)
        self.X = (self.X + dt * (self.v0 * self.omega + F + drift)) % L
        # (6) memory, exact relaxation
        aa = np.exp(-dt / self.tau_m)
        self.Mem = aa * self.Mem + (1 - aa) * rho_i
        # (7) deposit uptake at the new positions, (8) advance the field
        U = deposit(self.X, L, M)
        self.field.step(dt, U)
        self.t += dt

    def run(self, nsteps, record_every=0, metrics=None):
        hist = {"t": []}
        if metrics:
            for k in metrics:
                hist[k] = []
        for n in range(nsteps):
            self.step()
            if record_every and (n % record_every == 0 or n == nsteps - 1):
                hist["t"].append(self.t)
                if metrics:
                    for k, f in metrics.items():
                        hist[k].append(f(self))
        return hist


# ------------------------------------------------------------- diagnostics
def R_c(sim):
    d = min_image(sim.X - sim.C[None, :], sim.L)
    return float(np.sqrt((d * d).sum(1)).mean())


def R_unif(L):
    """Mean distance from a uniform point on the torus to a fixed point."""
    n = 2000
    xs = (np.arange(n) + 0.5) * L / n
    GX, GY = np.meshgrid(xs, xs, indexing="ij")
    dx = min_image(GX - 0.5 * L, L)
    dy = min_image(GY - 0.5 * L, L)
    return float(np.sqrt(dx ** 2 + dy ** 2).mean())


def aggregation_index(sim, r_a=0.05):
    N, L = sim.N, sim.L
    exp_rand = (N - 1) * np.pi * r_a ** 2 / (L * L)
    z = min_image(sim.X[:, None, :] - sim.X[None, :, :], L)
    d = np.sqrt((z * z).sum(2))
    cnt = np.sum((d <= r_a) & (d > 0)) / N
    return float(cnt / max(exp_rand, 1e-12))


def nutrient_mean(sim):
    return float(sim.field.rho.mean())


def density_field(sim):
    return deposit(sim.X, sim.L, sim.M)
