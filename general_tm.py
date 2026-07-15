"""
EXPLICIT transfer matrix for any periodic flat-foldable triangular-grid pattern.
Gives EXACT per-row eigenvalue Lambda_L (numpy), no convergence guessing.

Geometry (matches validated tri_transfer): vertex (r,j), j periodic mod L.
  dir0=(1,0)=d0(r,j) DOWN ; dir5=(1,-1)=d1(r,j) DOWN
  dir3=(-1,0)=up d0(r-1,j) ; dir2=(-1,1)=up d1(r-1,j+1)
  dir1=(0,1)=h(r,j) same-row right ; dir4=(0,-1)=h(r,j-1) same-row left
State between rows = MV of creased DOWN edges. Row transfer built by a ring-DP over j
(carry = h(r,j) MV). Full transfer over a rows = product; Lambda_L = top eigenvalue^(1/a).
"""
import itertools, numpy as np
from pattern_atlas import VALID, vertex_dirs, DIRDEG

def down_slots(creased, a, b, t, L):
    slots = []
    for j in range(L):
        S = vertex_dirs(t, j, creased, a, b)
        if 0 in S: slots.append(('d0', j))
        if 5 in S: slots.append(('d1', j))
    return slots

def valid_key_and_order(S):
    """VALID table keyed by MV in ascending dir order; return sorted dir list."""
    return sorted(S)

def row_transfer(creased, a, b, t, L):
    """Return (in_slots, out_slots, T) where T[out_index][in_index] = weight."""
    in_slots = down_slots(creased, a, b, (t - 1) % a, L)
    out_slots = down_slots(creased, a, b, t, L)
    in_index = {s: k for k, s in enumerate(in_slots)}
    out_index = {s: k for k, s in enumerate(out_slots)}
    Din, Dout = 2 ** len(in_slots), 2 ** len(out_slots)
    T = np.zeros((Dout, Din))

    # precompute per-vertex creased dir set and its VALID table
    vS = [vertex_dirs(t, j, creased, a, b) for j in range(L)]
    vtab = [VALID[frozenset(S)] for S in vS]
    vord = [sorted(S) for S in vS]

    def bits(state, slots):
        return {slots[k]: (1 if (state >> k) & 1 else -1) for k in range(len(slots))}

    for in_state in range(Din):
        inb = bits(in_state, in_slots)
        # ring DP over j; carry = h(t,j) MV (only matters if creased). We fix the seam
        # h(t,L-1) value, then DP left-to-right, requiring match at wrap.
        # h(t,j) creased?  dir1 in vS[j]
        hcreased = [1 in vS[j] for j in range(L)]
        seam_vals = [(-1, 1)] if hcreased[L - 1] else [(None,)]
        out_acc = {}
        for seam in ([-1, 1] if hcreased[L - 1] else [None]):
            # dp[carry_val or None] = {out_state_partial(dict of out slots filled): weight}
            # carry = h(t, j-1) coming into vertex j ; for j=0 it's the seam (h(t,L-1))
            dp = {seam: {(): 1}}
            for j in range(L):
                S = vS[j]; tab = vtab[j]; order = vord[j]
                # incident MV variables by dir:
                #  dir0 -> d0(t,j) DOWN ; dir5 -> d1(t,j) DOWN
                #  dir3 -> inb[('d0',j)] ; dir2 -> inb[('d1',(j+1)%L)]
                #  dir1 -> h(t,j) (new carry) ; dir4 -> h(t,j-1) (current carry)
                ndp = {}
                for carry, partials in dp.items():
                    # enumerate free vars at this vertex: d0,d1 (down, if creased), h_right (if creased)
                    d0opts = [(-1, 1)] if 0 in S else [(None,)]
                    d1opts = [(-1, 1)] if 5 in S else [(None,)]
                    hropts = [-1, 1] if 1 in S else [None]
                    for d0v in ([-1, 1] if 0 in S else [None]):
                        for d1v in ([-1, 1] if 5 in S else [None]):
                            for hrv in ([-1, 1] if 1 in S else [None]):
                                # assemble MV per dir present in S, in sorted order
                                mv = []
                                for d in order:
                                    if d == 0: mv.append(d0v)
                                    elif d == 5: mv.append(d1v)
                                    elif d == 3: mv.append(inb[('d0', j)])
                                    elif d == 2: mv.append(inb[('d1', (j + 1) % L)])
                                    elif d == 1: mv.append(hrv)
                                    elif d == 4: mv.append(carry)
                                if tuple(mv) not in tab:
                                    continue
                                w = tab[tuple(mv)]
                                for pstate, pw in partials.items():
                                    add = []
                                    if 0 in S: add.append((('d0', j), d0v))
                                    if 5 in S: add.append((('d1', j), d1v))
                                    newp = pstate + tuple(add)
                                    newcarry = hrv  # becomes h(t,j) carry for j+1
                                    key = newcarry
                                    ndp.setdefault(key, {})
                                    ndp[key][newp] = ndp[key].get(newp, 0) + w * pw
                dp = ndp
            # after processing all j, require final carry == seam (ring closes)
            final = dp.get(seam, {})
            for pstate, w in final.items():
                # pstate is list of ((slot),val); build out_state int
                d = dict(pstate)
                os = 0
                for k, s in enumerate(out_slots):
                    if d.get(s, 1) == 1:
                        os |= (1 << k)
                out_acc[os] = out_acc.get(os, 0) + w
        for os, w in out_acc.items():
            T[os][in_state] += w
    return in_slots, out_slots, T

def lambda_L(creased, a, b, L):
    mats = []
    for t in range(a):
        _, _, T = row_transfer(creased, a, b, t, L)
        mats.append(T)
    M = mats[0]
    for t in range(1, a):
        M = mats[t] @ M
    ev = np.linalg.eigvals(M)
    lam_super = max(abs(ev))
    return lam_super ** (1.0 / a)   # per single i-row

if __name__ == "__main__":
    from pattern_atlas import enumerate_patterns
    print("VALIDATE triangle lattice (tri_transfer gave Lambda_L=15, 56.22, 210.7):")
    pats, _ = enumerate_patterns(1, 1)
    tri = [c for c, comp, ty in pats if all(len(s) == 6 for s in comp.values())][0]
    for L in (1, 2, 3):
        print(f"   L={L}: Lambda_L = {lambda_L(tri,1,1,L):.5f}")
    print("\nVALIDATE bird's-foot (bulk torus gave 3, 4, then non-integer):")
    p2, _ = enumerate_patterns(2, 1)
    bf = [c for c, comp, ty in p2 if all(len(s) == 4 for s in comp.values())][0]
    for L in (2, 3, 4, 5, 6):
        lam = lambda_L(bf, 2, 1, L)
        print(f"   L={L}: Lambda_L = {lam:.6f}  g_L = {lam**(1.0/L):.6f}")
