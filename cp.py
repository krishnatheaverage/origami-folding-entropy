"""
Crease-pattern engine: build patches, attach per-vertex local-flat-foldability
constraints (from vertex.py), and count global valid MV assignments exactly via
variable (bucket) elimination.  Also an independent proper-q-coloring counter for
validation against Ginepro-Hull (Miura <-> 3-colorings).
"""
import math
from itertools import product
from functools import reduce
from vertex import local_flat_foldable, kawasaki_ok

# ----------------------------------------------------------------------------
# Generic exact counter: factors over {+1,-1} crease variables, bucket elimination
# ----------------------------------------------------------------------------
class Factor:
    __slots__ = ("vars", "table")
    def __init__(self, vars_, table):
        self.vars = list(vars_)           # ordered variable ids
        self.table = table                # dict: tuple(values) -> count (int)

def join_sumout(factors, var):
    """Multiply all factors mentioning `var`, then sum `var` out. Returns new factor list."""
    involved = [f for f in factors if var in f.vars]
    rest = [f for f in factors if var not in f.vars]
    if not involved:
        return factors
    # union of vars
    allvars = []
    for f in involved:
        for v in f.vars:
            if v not in allvars:
                allvars.append(v)
    idx = {v: i for i, v in enumerate(allvars)}
    new_table = {}
    # iterate over assignments of allvars
    doms = [(1, -1)] * len(allvars)
    for assign in product(*doms):
        prod = 1
        ok = True
        for f in involved:
            key = tuple(assign[idx[v]] for v in f.vars)
            c = f.table.get(key, 0)
            if c == 0:
                ok = False
                break
            prod *= c
        if not ok:
            continue
        # sum out var: key without var
        outvars = [v for v in allvars if v != var]
        outkey = tuple(assign[idx[v]] for v in outvars)
        new_table[outkey] = new_table.get(outkey, 0) + prod
    outvars = [v for v in allvars if v != var]
    rest.append(Factor(outvars, new_table))
    return rest

def count_assignments(factors, all_vars, order=None):
    """Exact count = sum over all var assignments of product of factor values."""
    factors = list(factors)
    if order is None:
        order = list(all_vars)
    for var in order:
        factors = join_sumout(factors, var)
    # multiply remaining scalar factors
    total = 1
    for f in factors:
        # f should now have no vars (fully eliminated) -> table {(): count}
        if f.vars:
            # variables that never appeared in any constraint: free, multiply by 2 each
            total *= 2 ** len(f.vars)
            # and sum its table
            total *= sum(f.table.values()) if f.table else 1
        else:
            total *= f.table.get((), 1)
    return total

# ----------------------------------------------------------------------------
# Build a crease pattern as a list of interior-vertex constraints
# ----------------------------------------------------------------------------
class CreasePattern:
    def __init__(self):
        self.creases = set()          # crease ids
        self.vertices = []            # list of (ordered crease-id list, angles list)

    def add_vertex(self, crease_ids, angles):
        assert len(crease_ids) == len(angles)
        assert kawasaki_ok(angles), f"vertex fails Kawasaki: {[math.degrees(a) for a in angles]}"
        self.creases.update(crease_ids)
        self.vertices.append((list(crease_ids), list(angles)))

    def vertex_factor(self, crease_ids, angles):
        """Factor whose table lists all locally-flat-foldable MV tuples for this vertex."""
        d = len(crease_ids)
        table = {}
        for mv in product((1, -1), repeat=d):
            if local_flat_foldable(angles, list(mv)):
                table[mv] = 1
        return Factor(crease_ids, table)

    def count_valid(self, order=None):
        factors = [self.vertex_factor(cids, ang) for (cids, ang) in self.vertices]
        return count_assignments(factors, list(self.creases), order)

# ----------------------------------------------------------------------------
# Miura patch: R x C grid of interior vertices, degree 4 each
# ----------------------------------------------------------------------------
def miura_pattern(R, C, alpha_deg=60, alternating=False):
    """Interior vertices at (i,j), i in 0..R-1, j in 0..C-1.
       Cyclic crease order at each vertex: (right, up, left, down).
       Straight horizontal line through vertex => sectors (a, pi-a, pi-a, a) [collinear].
       alternating=True uses (a, pi-a, a, pi-a)."""
    a = math.radians(alpha_deg)
    pi = math.pi
    if alternating:
        ang = [a, pi - a, a, pi - a]
    else:
        ang = [a, pi - a, pi - a, a]
    cp = CreasePattern()
    def H(i, j0, j1):
        return ("H", i, min(j0, j1), max(j0, j1))
    def V(i0, i1, j):
        return ("V", min(i0, i1), max(i0, i1), j)
    for i in range(R):
        for j in range(C):
            right = H(i, j, j + 1)
            left = H(i, j - 1, j)
            up = V(i - 1, i, j)
            down = V(i, i + 1, j)
            cp.add_vertex([right, up, left, down], ang)
    return cp

# ----------------------------------------------------------------------------
# Independent proper q-coloring counter on a graph (for validation)
# ----------------------------------------------------------------------------
def count_qcolorings(nodes, edges, q=3):
    """Count proper q-colorings via bucket elimination over node variables (domain 0..q-1)."""
    # factor per edge: allowed pairs where colors differ
    node_list = list(nodes)
    # represent as factors with domain q
    class QF:
        def __init__(self, vars_, table):
            self.vars = list(vars_); self.table = table
    factors = []
    for (u, v) in edges:
        table = {}
        for cu in range(q):
            for cv in range(q):
                if cu != cv:
                    table[(cu, cv)] = 1
        factors.append(QF([u, v], table))
    # also ensure isolated nodes counted: add unary factor of all-ones
    for n in node_list:
        factors.append(QF([n], {(c,): 1 for c in range(q)}))
    def qjoin(factors, var):
        involved = [f for f in factors if var in f.vars]
        rest = [f for f in factors if var not in f.vars]
        if not involved: return factors
        allvars = []
        for f in involved:
            for v in f.vars:
                if v not in allvars: allvars.append(v)
        idx = {v: i for i, v in enumerate(allvars)}
        new_table = {}
        for assign in product(range(q), repeat=len(allvars)):
            prod = 1; ok = True
            for f in involved:
                key = tuple(assign[idx[v]] for v in f.vars)
                c = f.table.get(key, 0)
                if c == 0: ok = False; break
                prod *= c
            if not ok: continue
            outvars = [v for v in allvars if v != var]
            outkey = tuple(assign[idx[v]] for v in outvars)
            new_table[outkey] = new_table.get(outkey, 0) + prod
        outvars = [v for v in allvars if v != var]
        rest.append(QF(outvars, new_table))
        return rest
    for var in node_list:
        factors = qjoin(factors, var)
    total = 1
    for f in factors:
        total *= f.table.get((), 1)
    return total

def grid_graph(R, C):
    nodes = [(i, j) for i in range(R) for j in range(C)]
    edges = []
    for i in range(R):
        for j in range(C):
            if j + 1 < C: edges.append(((i, j), (i, j + 1)))
            if i + 1 < R: edges.append(((i, j), (i + 1, j)))
    return nodes, edges


def brute_count_valid(cp):
    """Independent O(2^E) brute-force cross-check of count_valid()."""
    creases = sorted(cp.creases, key=lambda x: (str(type(x)), str(x)))
    idx = {c: i for i, c in enumerate(creases)}
    E = len(creases)
    total = 0
    for bits in product((1, -1), repeat=E):
        ok = True
        for (cids, ang) in cp.vertices:
            mv = [bits[idx[c]] for c in cids]
            if not local_flat_foldable(ang, mv):
                ok = False
                break
        if ok:
            total += 1
    return total


if __name__ == "__main__":
    print("=== cross-check bucket-elimination vs brute force (Miura) ===")
    for (R, C) in [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3)]:
        cp = miura_pattern(R, C, alpha_deg=60)
        be = cp.count_valid()
        bf = brute_count_valid(cp)
        print(f"Miura {R}x{C}: bucket={be}  brute={bf}  {'OK' if be==bf else 'MISMATCH!!'}")

    print("\n=== Miura MV counts (collinear vertex type) ===")
    rows = {}
    for R in range(1, 5):
        for C in range(1, 6):
            cp = miura_pattern(R, C, alpha_deg=60)
            mv = cp.count_valid()
            rows[(R, C)] = mv
            print(f"Miura interior {R}x{C}: valid MV = {mv}")

    print("\n=== proper 3-colorings of grid graphs P_R x P_C ===")
    for R in range(1, 5):
        line = []
        for C in range(1, 6):
            nodes, edges = grid_graph(R, C)
            line.append(count_qcolorings(nodes, edges, 3))
        print(f"3-col grid {R}x*: {line}")

    print("\n=== bijection probes (Miura MV vs 2*3colorings of grids) ===")
    for (R, C) in [(1,1),(1,2),(2,2),(2,3),(3,3)]:
        nodes, edges = grid_graph(R, C)
        col = count_qcolorings(nodes, edges, 3)
        print(f"{R}x{C}: MV={rows[(R,C)]}  3col(P{R}xP{C})={col}  MV/3col={rows[(R,C)]/col:.4f}  MV/2={rows[(R,C)]/2}")

    print("\n=== per-vertex growth ratios (square Miura NxN) ===")
    import math
    prev = None
    for N in range(1, 5):
        cp = miura_pattern(N, N, alpha_deg=60)
        c = cp.count_valid()
        gv = c ** (1.0 / (N * N))
        print(f"Miura {N}x{N}: count={c}  count^(1/N^2)={gv:.5f}")
