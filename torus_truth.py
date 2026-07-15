"""
INDEPENDENT GROUND TRUTH: exact local-flat-foldable MV counts on a fully-periodic
Lx x Ly torus, by bucket elimination (no transfer matrix). Used to (1) arbitrate the
tri_transfer vs general_tm eigenvalue discrepancy, (2) extract mu rigorously, and
(3) emit clean integer sequences for OEIS.

Uses the SAME VALID tables + edge-class conventions as pattern_atlas/general_tm, so a
match validates the transfer matrix exactly.
"""
import numpy as np
from cp import Factor, count_assignments
from pattern_atlas import VALID, vertex_dirs, enumerate_patterns
from general_tm import row_transfer

# incident (dir -> crease-variable) map on the torus, matching pattern_atlas FWD/BACK
def incident_vars(i, j, Lx, Ly):
    return {
        0: (i % Lx, j % Ly, 0),
        1: (i % Lx, j % Ly, 1),
        2: (i % Lx, j % Ly, 2),
        3: ((i - 1) % Lx, j % Ly, 0),
        4: (i % Lx, (j - 1) % Ly, 1),
        5: ((i + 1) % Lx, (j - 1) % Ly, 2),
    }

def exact_torus(creased, a, b, Lx, Ly):
    """Exact MV count on Lx x Ly torus tiling the a x b creased unit cell."""
    assert Lx % a == 0 and Ly % b == 0
    factors = []
    allvars = set()
    for i in range(Lx):
        for j in range(Ly):
            S = vertex_dirs(i % a, j % b, creased, a, b)   # creased dirs at this vertex
            if not S:
                continue
            inc = incident_vars(i, j, Lx, Ly)
            order = sorted(S)                               # matches VALID key order
            vars_ = [inc[d] for d in order]
            factors.append(Factor(vars_, VALID[frozenset(S)]))
            allvars.update(vars_)
    if not factors:
        return 1
    # elimination order: by (col, row, class) tends to keep buckets small
    order = sorted(allvars, key=lambda v: (v[1], v[0], v[2]))
    return count_assignments(factors, list(allvars), order)

def trace_pow(M, p):
    ev = np.linalg.eigvals(M)
    return float(np.real(np.sum(ev ** p)))

if __name__ == "__main__":
    print("="*70)
    print("TRIANGLE LATTICE (all deg-6 hub) — exact torus counts vs Tr(M^Lx)")
    print("="*70)
    pats, _ = enumerate_patterns(1, 1)
    tri = [c for c, comp, ty in pats if all(len(s) == 6 for s in comp.values())][0]
    # transfer matrices per circumference Ly
    Mcache = {}
    for Ly in (2, 3, 4):
        _, _, M = row_transfer(tri, 1, 1, 0, Ly)
        Mcache[Ly] = M
    print(f"{'Lx':>3} {'Ly':>3} {'exact torus':>16} {'Tr(M_Ly^Lx)':>16}  match")
    for Ly in (2, 3, 4):
        for Lx in (2, 3, 4):
            ex = exact_torus(tri, 1, 1, Lx, Ly)
            tr = trace_pow(Mcache[Ly], Lx)
            ok = abs(ex - tr) < 0.5
            print(f"{Lx:>3} {Ly:>3} {ex:>16d} {tr:>16.1f}  {'OK' if ok else 'MISMATCH'}")
    print("\nmu from largest-circumference eigenvalue lambda_max(M_Ly)^(1/Ly):")
    for Ly in (2, 3, 4):
        lam = max(abs(np.linalg.eigvals(Mcache[Ly])))
        print(f"   Ly={Ly}: lambda_max={lam:.5f}  mu_est=lambda^(1/Ly)={lam**(1.0/Ly):.6f}")
