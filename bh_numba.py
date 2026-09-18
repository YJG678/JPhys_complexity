"""Compiled Barnes-Hut long-range force (2D) using Numba.

The model is unchanged: unit-weight particles on the torus [0,L)^2, monopole
(centroid) approximation with opening ratio s/d < theta, mean-field 1/N scaling.
Only the implementation differs from hybrid_model.long_range_barnes_hut: the
quadtree is stored in flat NumPy arrays, built by a jitted routine, and the
force is evaluated with an explicit stack in a parallel jitted loop. This removes
the pure-Python bottleneck and lets the scaling benchmark reach N ~ 1e6.

Validation (test_numba_bh) checks it against the exact direct summation and the
reference pure-Python tree.
"""
import numpy as np
from numba import njit, prange


# Node arrays (Structure-of-Arrays):
#   ox, oy : node origin (lower corner)
#   sz     : node side length
#   cnt    : number of bodies in subtree
#   cxs,cys: sum of body coordinates in subtree (centroid = sum/cnt)
#   ch     : (nnodes,4) child node indices, -1 if absent
#   leaf   : body index if node is a single-body leaf, else -1
@njit(cache=True)
def _octant(ox, oy, sz, x, y):
    half = 0.5 * sz
    bx = 1 if x >= ox + half else 0
    by = 1 if y >= oy + half else 0
    return by * 2 + bx, ox + bx * half, oy + by * half, half


@njit(cache=True)
def build_tree(X, L, max_nodes, max_depth):
    N = X.shape[0]
    ox = np.empty(max_nodes); oy = np.empty(max_nodes); sz = np.empty(max_nodes)
    cnt = np.zeros(max_nodes, np.int64)
    cxs = np.zeros(max_nodes); cys = np.zeros(max_nodes)
    ch = np.full((max_nodes, 4), -1, np.int64)
    leaf = np.full(max_nodes, -1, np.int64)

    ox[0] = 0.0; oy[0] = 0.0; sz[0] = L
    nfree = 1

    for p in range(N):
        x = X[p, 0]; y = X[p, 1]
        n = 0
        depth = 0
        while True:
            if cnt[n] == 0 and leaf[n] == -1:
                # empty node -> leaf holding p
                leaf[n] = p; cnt[n] = 1; cxs[n] = x; cys[n] = y
                break
            if leaf[n] != -1:
                # occupied leaf: push the resident body q into a child
                q = leaf[n]; leaf[n] = -1
                xq = X[q, 0]; yq = X[q, 1]
                if depth >= max_depth:
                    # merge: keep both in this node's aggregate (do not split)
                    cnt[n] += 1; cxs[n] += x; cys[n] += y
                    break
                cq, cqx, cqy, cqs = _octant(ox[n], oy[n], sz[n], xq, yq)
                nc = nfree; nfree += 1
                ox[nc] = cqx; oy[nc] = cqy; sz[nc] = cqs
                leaf[nc] = q; cnt[nc] = 1; cxs[nc] = xq; cys[nc] = yq
                ch[n, cq] = nc
            # internal node: add p to aggregate and descend
            cnt[n] += 1; cxs[n] += x; cys[n] += y
            cp, cpx, cpy, cps = _octant(ox[n], oy[n], sz[n], x, y)
            if ch[n, cp] == -1:
                nc = nfree; nfree += 1
                ox[nc] = cpx; oy[nc] = cpy; sz[nc] = cps
                leaf[nc] = p; cnt[nc] = 1; cxs[nc] = x; cys[nc] = y
                ch[n, cp] = nc          # link the new leaf to its parent
                break
            n = ch[n, cp]
            depth += 1
            if depth > max_depth + 8:      # hard safety stop
                cnt[n] += 1; cxs[n] += x; cys[n] += y
                break
    return ox, oy, sz, cnt, cxs, cys, ch, leaf, nfree


@njit(cache=True, parallel=True)
def eval_forces(X, L, eps_l, reg, theta,
                ox, oy, sz, cnt, cxs, cys, ch, leaf, nfree):
    N = X.shape[0]
    F = np.zeros((N, 2))
    theta2 = theta * theta
    reg2 = reg * reg
    for i in prange(N):
        xi = X[i, 0]; yi = X[i, 1]
        fx = 0.0; fy = 0.0
        stack = np.empty(64 + 4 * 30, np.int64)   # ample explicit stack
        sp = 0
        stack[sp] = 0; sp += 1
        while sp > 0:
            sp -= 1
            n = stack[sp]
            c = cnt[n]
            if c == 0:
                continue
            ccx = cxs[n] / c
            ccy = cys[n] / c
            dx = xi - ccx; dx -= L * np.round(dx / L)
            dy = yi - ccy; dy -= L * np.round(dy / L)
            d2 = dx * dx + dy * dy
            s = sz[n]
            is_leaf = leaf[n] != -1
            if is_leaf and leaf[n] == i:
                continue                              # skip self
            # opening test  s/d < theta   <=>   s^2 < theta^2 d2
            if is_leaf or (s * s < theta2 * d2):
                if d2 > 1e-18:
                    coeff = -eps_l * c / (d2 + reg2)
                    fx += coeff * dx; fy += coeff * dy
            else:
                for k in range(4):
                    cc = ch[n, k]
                    if cc != -1:
                        stack[sp] = cc; sp += 1
        F[i, 0] = fx / N
        F[i, 1] = fy / N
    return F


def long_range_bh_numba(X, L, eps_l, reg, theta, max_depth=28):
    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[0]
    max_nodes = 4 * N + 64
    tree = build_tree(X, L, max_nodes, max_depth)
    return eval_forces(X, L, eps_l, reg, theta, *tree)


@njit(cache=True, parallel=True)
def count_interactions(X, L, theta, ox, oy, sz, cnt, cxs, cys, ch, leaf, nfree):
    """Total number of monopole/leaf force terms evaluated (algorithmic work)."""
    N = X.shape[0]
    counts = np.zeros(N, np.int64)
    theta2 = theta * theta
    for i in prange(N):
        xi = X[i, 0]; yi = X[i, 1]
        c_i = 0
        stack = np.empty(64 + 4 * 30, np.int64)
        sp = 0; stack[sp] = 0; sp += 1
        while sp > 0:
            sp -= 1; n = stack[sp]; c = cnt[n]
            if c == 0:
                continue
            ccx = cxs[n] / c; ccy = cys[n] / c
            dx = xi - ccx; dx -= L * np.round(dx / L)
            dy = yi - ccy; dy -= L * np.round(dy / L)
            d2 = dx * dx + dy * dy; s = sz[n]
            is_leaf = leaf[n] != -1
            if is_leaf and leaf[n] == i:
                continue
            if is_leaf or (s * s < theta2 * d2):
                if d2 > 1e-18:
                    c_i += 1
            else:
                for k in range(4):
                    cc = ch[n, k]
                    if cc != -1:
                        stack[sp] = cc; sp += 1
        counts[i] = c_i
    return counts


# ----------------------------------------------------------------------
# Compiled (Numba) cell-list short-range force. Binning is done in NumPy;
# the neighbour loop is jitted and parallel. Exact for compactly supported
# kernels when the cell width h = L/ncell >= r_c.
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _celllist_force(X, L, eps_s, r_c, ncell, h, order, cell_start, cell_end, reg_s):
    N = X.shape[0]
    F = np.zeros((N, 2))
    for i in prange(N):
        xi = X[i, 0]; yi = X[i, 1]
        cx = int(xi / h) % ncell
        cy = int(yi / h) % ncell
        fx = 0.0; fy = 0.0
        for ox in range(-1, 2):
            nx = (cx + ox) % ncell
            for oy in range(-1, 2):
                ny = (cy + oy) % ncell
                c = ny * ncell + nx
                for t in range(cell_start[c], cell_end[c]):
                    j = order[t]
                    if j == i:
                        continue
                    dx = xi - X[j, 0]; dx -= L * np.round(dx / L)
                    dy = yi - X[j, 1]; dy -= L * np.round(dy / L)
                    d = np.sqrt(dx * dx + dy * dy)
                    if 0.0 < d <= r_c:
                        den = np.sqrt(d * d + reg_s * reg_s) if reg_s > 0.0 else d
                        m = eps_s * (1.0 - d / r_c) / den
                        fx += m * dx; fy += m * dy
        F[i, 0] = fx; F[i, 1] = fy
    return F / N


def short_range_celllist_numba(X, L, eps_s, r_c, reg_s=0.0):
    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[0]
    ncell = max(1, int(np.floor(L / r_c)))
    h = L / ncell
    cix = (np.floor(X[:, 0] / h).astype(np.int64)) % ncell
    ciy = (np.floor(X[:, 1] / h).astype(np.int64)) % ncell
    cell_of = ciy * ncell + cix
    order = np.argsort(cell_of, kind="stable").astype(np.int64)
    cs = cell_of[order]
    ncells = ncell * ncell
    cell_start = np.searchsorted(cs, np.arange(ncells)).astype(np.int64)
    cell_end = np.searchsorted(cs, np.arange(ncells) + 1).astype(np.int64)
    return _celllist_force(X, L, eps_s, r_c, ncell, h, order, cell_start, cell_end, reg_s)
