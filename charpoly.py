"""
EXACT characteristic polynomials of the bulk row-transfer matrices M_L (integer
entries), to decide whether the dominant eigenvalue Lambda_L has a clean closed form
or the '3,4,...' integer pattern is a small-size / boundary artifact.
"""
import numpy as np, sympy as sp
from pattern_atlas import enumerate_patterns
from general_tm import row_transfer

def intM(creased, a, b, L):
    # product of the a per-i-row transfers -> super-row transfer (integer)
    mats = [row_transfer(creased, a, b, t, L)[2] for t in range(a)]
    M = mats[0]
    for t in range(1, a):
        M = mats[t] @ M
    assert np.max(np.abs(M - np.rint(M))) < 1e-6, "non-integer transfer entry!"
    n = M.shape[0]
    return sp.Matrix(n, n, [int(round(M[r, c])) for r in range(n) for c in range(n)])

def analyze(name, creased, a, b, Ls):
    print("="*66); print(name); print("="*66)
    lam = sp.symbols('lam')
    for L in Ls:
        M = intM(creased, a, b, L)
        cp = M.charpoly(lam)                                  # exact integer char poly
        evs = np.linalg.eigvals(np.array(M.tolist(), dtype=float))
        dom = max(evs, key=abs)
        facs = sp.factor_list(cp.as_expr())[1]
        dom_poly, dom_deg = None, None
        for f, mult in facs:
            P = sp.Poly(f, lam)
            if P.degree() < 1:
                continue
            coeffs = [float(c) for c in P.all_coeffs()]   # integer coeffs already
            for rt in np.roots(coeffs):
                if abs(rt - dom) < 1e-5:
                    dom_poly, dom_deg = P, P.degree()
        a_super = a  # super-row = a i-rows
        # per-i-row eigenvalue = dom^(1/a); per-vertex mu-estimate = dom^(1/(a*L))
        mu_est = abs(dom) ** (1.0 / (a_super * L))
        print(f"L={L}: dim={M.shape[0]}  Lambda_super={dom.real:.6f}  "
              f"per-i-row={abs(dom)**(1.0/a_super):.6f}  mu_est(^1/{a_super*L})={mu_est:.6f}")
        if dom_poly is not None:
            print(f"   exact min poly of Lambda_super (deg {dom_deg}): {dom_poly.as_expr()} = 0")
    print()

if __name__ == "__main__":
    pats1, _ = enumerate_patterns(1, 1)
    tri = [c for c, comp, ty in pats1 if all(len(s) == 6 for s in comp.values())][0]
    analyze("TRIANGLE LATTICE (all deg-6 hub)  [super = 1 i-row]", tri, 1, 1, (2, 3, 4))

    pats2, _ = enumerate_patterns(2, 1)
    bf = [c for c, comp, ty in pats2 if all(len(s) == 4 for s in comp.values())][0]
    analyze("BIRD'S-FOOT triangular (all deg-4)  [super = 2 i-rows]", bf, 2, 1, (2, 3, 4))
