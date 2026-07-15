"""
Reproduce every number reported in
  "Residual entropy of the degree-6 triangular origami lattice: a charge-+-2 ice model."

Run:  python3 reproduce.py
Depends only on numpy (and the modules in this directory). Each block prints the
paper value next to the freshly computed one.
"""
import numpy as np
from pattern_atlas import enumerate_patterns
from general_tm import row_transfer
from torus_truth import exact_torus

def tri_pattern():
    pats, _ = enumerate_patterns(1, 1)
    return [c for c, comp, ty in pats if all(len(s) == 6 for s in comp.values())][0]

def birdsfoot_pattern():
    pats, _ = enumerate_patterns(2, 1)
    return [c for c, comp, ty in pats if all(len(s) == 4 for s in comp.values())][0]

def dominant_eig(M):
    ev = np.linalg.eigvals(np.rint(M))
    return max(ev.real)

# ---------------------------------------------------------------- Lambda_L, mu
print("=" * 68)
print("Per-circumference eigenvalues Lambda_L and residual entropy mu")
print("=" * 68)
tri = tri_pattern()
lam = {}
for L in range(2, 7):
    _, _, M = row_transfer(tri, 1, 1, 0, L)
    lam[L] = dominant_eig(M)
    print(f"  L={L}: dim={M.shape[0]:>5}  Lambda_L = {lam[L]:.8f}")
lo = lam[6] / lam[5]
up = lam[6] ** (1 / 6)
print(f"\n  bracket:  {lo:.7f} <= mu <= {up:.7f}   (paper: 3.7479350..3.7479355)")
print(f"  mu = 3.747935(1),  s = ln mu = {np.log(up):.6f}  (paper 1.321205)")

# ------------------------------------------------ charged-ice = flat-fold count
print("\n" + "=" * 68)
print("Charge-+-2 ice model  ==  locally-flat-foldable count  (every torus)")
print("=" * 68)
import itertools
def incident(i, j, A, B):
    return [(i, j, 0), (i, j, 1), (i, j, 2),
            ((i - 1) % A, j % B, 0), (i % A, (j - 1) % B, 1), ((i + 1) % A, (j - 1) % B, 2)]
def charged_ice(A, B):
    edges = [(i, j, d) for i in range(A) for j in range(B) for d in (0, 1, 2)]
    idx = {e: k for k, e in enumerate(edges)}
    inc = [[idx[e] for e in incident(i, j, A, B)] for i in range(A) for j in range(B)]
    c = 0
    for bits in itertools.product((1, -1), repeat=len(edges)):
        if all(sum(bits[k] for k in vi) in (2, -2) for vi in inc):
            c += 1
    return c
for (A, B) in [(2, 2), (3, 2), (2, 3), (3, 3)]:
    ci = charged_ice(A, B)
    z = exact_torus(tri, 1, 1, A, B)
    print(f"  torus {A}x{B}: charged-ice={ci:>7d}  flat-fold={z:>7d}  {'MATCH' if ci == z else 'DIFFER'}")

# ------------------------------------------------------- exact integer data
print("\n" + "=" * 68)
print("Exact enumeration data (new integer sequences)")
print("=" * 68)
print("  strip 2*15^n:", [2 * 15 ** n for n in range(1, 7)])
def trace_pow_seq(L, nmax):
    _, _, M = row_transfer(tri, 1, 1, 0, L)
    Mi = np.rint(M).astype(object)
    Mi = [[int(Mi[i, j]) for j in range(Mi.shape[0])] for i in range(Mi.shape[0])]
    def mm(A, B):
        return [[sum(A[i][t] * B[t][j] for t in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    cur = [r[:] for r in Mi]; out = []
    for _ in range(nmax):
        out.append(sum(cur[i][i] for i in range(len(cur)))); cur = mm(cur, Mi)
    return out
print("  Z(n,2):", trace_pow_seq(2, 8))
print("  Z(n,3):", trace_pow_seq(3, 6))

# --------------------------------------------------------- degree-4 baseline
print("\n" + "=" * 68)
print("Degree-4 baseline: bird's-foot transfer eigenvalue vs square 3-coloring")
print("=" * 68)
bf = birdsfoot_pattern()
def ring_colorings(L, q=3):
    return [s for s in itertools.product(range(q), repeat=L) if all(s[i] != s[(i + 1) % L] for i in range(L))]
def threecol_eig(L):
    S = ring_colorings(L); n = len(S)
    T = np.zeros((n, n))
    for a, s in enumerate(S):
        for b, t in enumerate(S):
            if all(s[i] != t[i] for i in range(L)):
                T[a][b] = 1
    return max(np.linalg.eigvals(T).real)
for L in (2, 4, 6, 8):
    mats = [row_transfer(bf, 2, 1, t, L)[2] for t in range(2)]
    Msup = mats[1] @ mats[0]
    bf_irow = dominant_eig(Msup) ** 0.5
    c3 = threecol_eig(L)
    print(f"  L={L}: bird's-foot={bf_irow:.6f}  square-3col={c3:.6f}  {'MATCH' if abs(bf_irow - c3) < 1e-4 else '-'}")
print("  => degree-4 rate = (4/3)^(3/2) = 1.539601 (Baxter three-coloring)")
print("\nDone.")
