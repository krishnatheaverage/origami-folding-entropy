"""
Extract the FACE structure of the all-bird's-foot triangular tessellation.

Ginepro-Hull (2014) / Assis (2018): valid MV assignments of a flat-foldable
degree-4 origami tessellation biject with proper 3-colorings of the crease
pattern's faces. So the folding-entropy growth constant equals the 3-coloring
entropy of the FACE-ADJACENCY graph. If that graph is a square grid, the
constant is Baxter's W = (4/3)^(3/2) -- explaining the observed universality.

This script builds the creased subgraph on a torus (only creased edges bound
faces; uncreased grid edges are interior flat paper), traces its faces using the
geometric embedding, and reports face sizes + the face-adjacency graph.
"""
import math
from collections import defaultdict
from pattern_atlas import enumerate_patterns

SQ3 = math.sqrt(3)
# forward directions on the triangular grid (dir index -> lattice offset)
FWD_OFF = {0: (1, 0), 1: (0, 1), 2: (-1, 1)}

def xy(i, j):
    return (i + 0.5 * j, SQ3 / 2.0 * j)

def creased_edges_on_torus(bf, a, b, N):
    """Return set of undirected creased edges as frozenset({(i,j),(i2,j2)}) on an
       (a*N) x (b*N) torus, plus the vertex set. Coordinates are kept as lattice
       (i,j) mod (A,B); we return both wrapped endpoints and the offset used."""
    A, B = a * N, b * N
    edges = set()
    for i in range(A):
        for j in range(B):
            for d, off in FWD_OFF.items():
                if bf[(i % a, j % b, d)]:
                    p = (i, j)
                    q = ((i + off[0]) % A, (j + off[1]) % B)
                    edges.add((p, q, d))   # keep direction class for embedding
    return edges, A, B

def build_incidence(edges, A, B):
    """adjacency: vertex -> list of (neighbor, angle) using the geometric
       embedding, with torus wrap resolved to the true local direction."""
    inc = defaultdict(list)
    # unit direction vectors for each dir class (and their reverses)
    dirvec = {0: (1.0, 0.0), 1: (0.5, SQ3/2.0), 2: (-0.5, SQ3/2.0)}
    for (p, q, d) in edges:
        vx, vy = dirvec[d]
        ang_pq = math.degrees(math.atan2(vy, vx)) % 360.0
        ang_qp = (ang_pq + 180.0) % 360.0
        inc[p].append((q, ang_pq))
        inc[q].append((p, ang_qp))
    # sort each vertex's incident edges ccw by angle
    for v in inc:
        inc[v].sort(key=lambda t: t[1])
    return inc

def trace_faces(inc):
    """Trace faces of the embedded graph via the next-edge (rotation system) rule.
       For directed edge (u->v), the next directed edge in the same face is
       (v->w) where w is the neighbor of v whose angle is the first one clockwise
       from the reverse direction (v->u). Standard planar face tracing."""
    # index neighbors per vertex with their angles
    nbr = {v: [q for (q, ang) in lst] for v, lst in inc.items()}
    angs = {v: {q: ang for (q, ang) in lst} for v, lst in inc.items()}
    visited = set()   # directed edges (u,v)
    faces = []
    for u in inc:
        for (v, _) in inc[u]:
            if (u, v) in visited:
                continue
            face = []
            cu, cv = u, v
            while (cu, cv) not in visited:
                visited.add((cu, cv))
                face.append((cu, cv))
                # at cv, reverse dir is angle of (cv->cu)
                rev = angs[cv][cu]
                # neighbors of cv sorted ccw; pick the one just CLOCKWISE of rev
                lst = inc[cv]
                # find rev index
                order = [q for (q, ang) in lst]
                angvals = [ang for (q, ang) in lst]
                # next clockwise = previous in ccw order relative to rev
                ridx = order.index(cu)
                nxt = order[(ridx - 1) % len(order)]
                cu, cv = cv, nxt
            faces.append(face)
    return faces

if __name__ == "__main__":
    pats2, _ = enumerate_patterns(2, 1)
    bf = [c for c, comp, ty in pats2 if all(len(s) == 4 for s in comp.values())][0]

    for N in (3, 4, 6):
        edges, A, B = creased_edges_on_torus(bf, 2, 1, N)
        inc = build_incidence(edges, A, B)
        nverts = len(inc)
        nedges = len(edges)
        faces = trace_faces(inc)
        # face sizes (each face traced once as a directed cycle)
        sizes = defaultdict(int)
        for f in faces:
            sizes[len(f)] += 1
        # Euler check on torus: V - E + F = 0
        V, E, F = nverts, nedges, len(faces)
        print(f"N={N}: torus {A}x{B}  V={V} E={E} F={F}  V-E+F={V-E+F} (torus expects 0)")
        print(f"   face-size histogram: {dict(sorted(sizes.items()))}")
        # average degree
        deg = [len(inc[v]) for v in inc]
        from collections import Counter
        print(f"   vertex-degree histogram: {dict(sorted(Counter(deg).items()))}")
        print()
