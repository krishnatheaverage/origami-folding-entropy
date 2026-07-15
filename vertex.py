"""
Single-vertex LOCAL flat-foldability predicate.

Ground-truth method (no reliance on memorized reduction rules):
  A flat vertex of degree d has d creases at angular directions theta_0<...<theta_{d-1}
  and d sectors with angles a_0..a_{d-1} (a_i between crease i and crease i+1), sum = 2*pi.
  Kawasaki (necessary for any flat fold): alternating sum of a_i = 0, i.e. the folded
  coordinates X_i close up.

  Fold the cone flat -> a CYCLIC 1-D folding.  Folded coordinate of crease i:
        X_0 = 0,  X_i = sum_{j<i} (-1)^j a_j .
  Kawasaki <=> X_d == X_0 == 0.  Sector i is the segment [X_i, X_{i+1}] living at some
  layer height; crease i (at position X_i) connects sector i-1 and sector i (cyclically).

  An MV assignment (m_i in {+1,-1}) is a VALID local flat fold iff there is a layer
  ordering (a permutation of the d sectors into distinct heights) such that:
    (1) taco-taco : two creases at the SAME folded position have nested or disjoint
        height-intervals, never interleaved (folds don't cross);
    (2) taco-tortilla : a sector strictly covering a crease's position must not sit at a
        height strictly between that crease's two layers (paper can't pierce a fold);
    (3) the MV signs are consistent with the heights (up to a single global M<->V flip,
        which is an exact symmetry of the valid set, so it does not affect COUNTS).

  d <= 6 for our patterns, so brute-forcing d! layer orders is instant and exact.
"""

from itertools import permutations
import math

EPS = 1e-9


def kawasaki_ok(angles):
    d = len(angles)
    if d % 2 != 0:
        return False
    s = 0.0
    for j, a in enumerate(angles):
        s += ((-1) ** j) * a
    return abs(s) < 1e-6 and abs(sum(angles) - 2 * math.pi) < 1e-6


def folded_positions(angles):
    d = len(angles)
    X = [0.0] * (d + 1)
    for i in range(1, d + 1):
        X[i] = X[i - 1] + ((-1) ** (i - 1)) * angles[i - 1]
    return X  # X[0..d], X[d] should == 0


def _valid_layering(order, X, mv, d):
    """order[seg] = height (int). Check (1),(2),(3) for one candidate layering.
       mv given up to global sign; we try both global signs (handled by caller)."""
    # crease i at position X[i], connects seg (i-1)%d and seg i
    creases = []
    for i in range(d):
        pos = X[i]
        s_lo = (i - 1) % d
        s_hi = i
        h0 = order[s_lo]
        h1 = order[s_hi]
        creases.append((pos, min(h0, h1), max(h0, h1), i))
    # (1) taco-taco: same position -> nested or disjoint
    for a in range(d):
        pa, la, ha, ia = creases[a]
        for b in range(a + 1, d):
            pb, lb, hb, ib = creases[b]
            if abs(pa - pb) < EPS:
                # intervals (la,ha) and (lb,hb): forbid interleave
                # interleave iff la<lb<ha<hb or lb<la<hb<ha
                if (la < lb < ha < hb) or (lb < la < hb < ha):
                    return False
    # (2) taco-tortilla: sector covering a crease position, height strictly between
    for seg in range(d):
        lo = min(X[seg], X[seg + 1])
        hi = max(X[seg], X[seg + 1])
        hseg = order[seg]
        for (pos, l, h, i) in creases:
            if lo - EPS < pos < hi + EPS and not (abs(pos - lo) < EPS or abs(pos - hi) < EPS):
                # crease strictly inside this sector's span
                if seg == i or seg == (i - 1) % d:
                    continue  # this sector owns the crease
                if l < hseg < h:
                    return False
    # (3) MV consistency up to global sign:
    # local fold direction at crease i determined by sign(h(seg_i)-h(seg_{i-1}))*(-1)^i
    got = []
    for i in range(d):
        s_lo = (i - 1) % d
        dirn = order[i] - order[s_lo]
        sign = 1 if (dirn * ((-1) ** i)) > 0 else -1
        got.append(sign)
    # compare to mv, allowing global flip
    if all(got[i] == mv[i] for i in range(d)):
        return True
    if all(got[i] == -mv[i] for i in range(d)):
        return True
    return False


def local_flat_foldable(angles, mv):
    """angles: sector angles (radians), len d even, Kawasaki assumed.
       mv: list of +-1 per crease (crease i before sector i). Returns True/False."""
    d = len(angles)
    # Maekawa necessary condition: |#M - #V| == 2
    if abs(sum(mv)) != 2:
        return False
    X = folded_positions(angles)
    for order in permutations(range(d)):
        if _valid_layering(list(order), X, mv, d):
            return True
    return False


def count_local_mv(angles):
    """Count valid MV assignments for a single isolated vertex."""
    d = len(angles)
    from itertools import product
    cnt = 0
    good = []
    for mv in product((1, -1), repeat=d):
        if local_flat_foldable(angles, list(mv)):
            cnt += 1
            good.append(mv)
    return cnt, good


if __name__ == "__main__":
    import math
    pi = math.pi
    # --- test: generic degree-4 flat-foldable vertex, unique smallest angle ---
    # angles satisfying Kawasaki: a0+a2 = a1+a3 = pi ; pick distinct
    deg = math.radians
    g4 = [deg(40), deg(80), deg(140), deg(100)]  # alt sum 40-80+140-100=0
    print("Kawasaki g4:", kawasaki_ok(g4))
    c, good = count_local_mv(g4)
    print("generic deg-4 valid MV count:", c)
    for m in good:
        print("   ", m)

    # --- Miura vertex: angles alpha, pi-alpha, pi-alpha, alpha (collinear pair) ---
    for adeg in (60, 70, 45):
        a = deg(adeg)
        miura = [a, pi - a, pi - a, a]
        print(f"\nMiura vertex alpha={adeg}: Kawasaki={kawasaki_ok(miura)}")
        c, good = count_local_mv(miura)
        print("   valid MV count:", c)
        for m in good:
            print("     ", m)

    # --- regular degree-6 vertex (all 60 deg) ---
    reg6 = [deg(60)] * 6
    print("\nregular deg-6 (all 60): Kawasaki:", kawasaki_ok(reg6))
    c, good = count_local_mv(reg6)
    print("   valid MV count:", c)
