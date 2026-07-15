"""
Systematic atlas of periodic flat-foldable crease patterns on the triangular grid.

A pattern = choice of creased edges, periodic with an a x b unit cell. Each vertex's
creased-direction subset must be a flat-foldable grid type (degree 0/2/4/6; the deg-4
is the unique bird's foot MV6, deg-6 the hub MV30 -- see gridtypes.py). For every valid
pattern we compute the folding-entropy growth constant of its local-flat-foldable MV
assignments via exact counts on tori of increasing size.
"""
import math, itertools
from vertex import local_flat_foldable, kawasaki_ok
from cp import Factor, count_assignments

DIRDEG = [0, 60, 120, 180, 240, 300]
# forward directions (edge owner) and the neighbor offset each points to:
FWD = {0: (1, 0), 1: (0, 1), 2: (-1, 1)}          # dirs 0,60,120
# backward incidence: dir 180 from neighbor (-1,0) fdir0; 240 from (0,-1) fdir1; 300 from (1,-1) fdir2
BACK = {3: ((-1, 0), 0), 4: ((0, -1), 1), 5: ((1, -1), 2)}

# ---- precompute valid vertex types and their MV factor (ordered by dir index) ----
def _mv_table(subset):
    ds = sorted(DIRDEG[i] for i in subset)
    n = len(ds)
    sec = [((ds[(k+1) % n] - ds[k]) % 360) or 360 for k in range(n)]
    ang = [math.radians(s) for s in sec]
    if n % 2 or not kawasaki_ok(ang):
        return None
    tbl = {}
    for mv in itertools.product((1, -1), repeat=n):
        if local_flat_foldable(ang, list(mv)):
            tbl[mv] = 1
    return tbl if tbl else None

VALID = {}  # frozenset(dir indices) -> mv table (order = sorted dir indices)
for r in (0, 2, 4, 6):
    for subset in itertools.combinations(range(6), r):
        if r == 0:
            VALID[frozenset()] = {(): 1}
            continue
        t = _mv_table(subset)
        if t is not None:
            VALID[frozenset(subset)] = t

def vertex_dirs(i, j, creased, a, b):
    """set of creased direction-indices at vertex (i,j)."""
    s = set()
    for d, off in FWD.items():
        if creased.get(((i) % a, (j) % b, d), False):
            s.add(d)
    for d, (off, fd) in BACK.items():
        ni, nj = (i + off[0]) % a, (j + off[1]) % b
        if creased.get((ni, nj, fd), False):
            s.add(d)
    return frozenset(s)

def pattern_valid(creased, a, b):
    comp = {}
    for i in range(a):
        for j in range(b):
            s = vertex_dirs(i, j, creased, a, b)
            if s not in VALID:
                return None
            comp[(i, j)] = s
    return comp

def enumerate_patterns(a, b):
    edges = [(i, j, d) for i in range(a) for j in range(b) for d in (0, 1, 2)]
    E = len(edges)
    results = []
    for bits in itertools.product((False, True), repeat=E):
        creased = {edges[k]: bits[k] for k in range(E)}
        comp = pattern_valid(creased, a, b)
        if comp is None:
            continue
        # skip all-uncreased trivial
        types = sorted(len(s) for s in comp.values())
        if set(types) == {0}:
            continue
        results.append((creased, comp, tuple(types)))
    return results, edges

# ---- growth constant: exact torus counts on (ma x nb) then per-vertex growth ----
def torus_count(creased, a, b, ma, mb):
    """count local-flat-foldable MV on an (a*ma) x (b*mb) torus of the pattern.
       Uses PRECOMPUTED per-type MV tables in VALID (keyed by dir-index subset),
       so no per-vertex re-enumeration of the layer-order predicate."""
    A, B = a * ma, b * mb
    on = set()
    creases = set()
    for i in range(A):
        for j in range(B):
            for d in (0, 1, 2):
                if creased[(i % a, j % b, d)]:
                    on.add((i, j, d)); creases.add((i, j, d))
    factors = []
    for i in range(A):
        for j in range(B):
            didx = []; cids = []
            for d, off in FWD.items():
                if (i, j, d) in on:
                    didx.append(d); cids.append((i, j, d))
            for d, (off, fd) in BACK.items():
                ni, nj = (i + off[0]) % A, (j + off[1]) % B
                if (ni, nj, fd) in on:
                    didx.append(d); cids.append((ni, nj, fd))
            if not didx:
                continue
            order = sorted(range(len(didx)), key=lambda k: DIRDEG[didx[k]])
            key = frozenset(didx)
            tbl = VALID[key]                       # precomputed, order = sorted dir index
            scids = [cids[k] for k in order]       # DIRDEG-sorted == dir-index-sorted (dirs monotone)
            factors.append(Factor(scids, tbl))
    return count_assignments(factors, list(creases), sorted(creases))

def growth_estimate(creased, a, b):
    """estimate per-vertex growth from torus ratios."""
    base = torus_count(creased, a, b, 1, 1)
    # grow in one direction
    c2 = torus_count(creased, a, b, 2, 1)
    c3 = torus_count(creased, a, b, 3, 1)
    per_cell_row = None
    if c2 > 0 and base > 0:
        per_cell_row = c3 / c2  # ~ lambda per (a rows x b) block
    nv = a * b
    return base, c2, c3, per_cell_row

if __name__ == "__main__":
    print("Valid vertex types loaded:", {len(s): 1 for s in VALID}.keys(), "distinct subsets:", len(VALID))
    for (a, b) in [(1, 1), (2, 1), (2, 2)]:
        pats, edges = enumerate_patterns(a, b)
        # dedup by type-composition signature
        sigs = {}
        for creased, comp, types in pats:
            from collections import Counter
            sig = tuple(sorted(Counter(len(s) for s in comp.values()).items()))
            sigs.setdefault(sig, []).append(creased)
        print(f"\n=== unit cell {a}x{b}: {len(pats)} valid patterns, {len(sigs)} type-signatures ===")
        for sig, lst in sorted(sigs.items()):
            comp_desc = ", ".join(f"{cnt}x deg{deg}" for deg, cnt in sig)
            print(f"  [{comp_desc}]  ({len(lst)} patterns)")
