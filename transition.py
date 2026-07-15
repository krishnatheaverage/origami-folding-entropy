"""
(c) Phase-transition test for the triangle-lattice folding model. Weight mountains by
fugacity y; compute per-vertex free energy f_L(y)=ln Lambda_L(y)/L and the
"specific heat" chi_L(y)=d^2 f/d(ln y)^2. If chi develops a growing/diverging peak as
circumference L increases, there is a transition; if chi stays bounded and L-independent,
the folding free energy is analytic (no transition). y<->1/y (M<->V) symmetry => test at
and near y=1.
"""
import math, numpy as np
from pattern_atlas import enumerate_patterns
from general_cylinder import cylinder_count

def f_L(creased, a, b, L, y, Hmax=12):
    Hs = list(range(2*a, Hmax+1, a))
    Ns = [cylinder_count(creased, a, b, L, H, y) for H in Hs]
    lam = (Ns[-1]/Ns[-2])**(1.0/(Hs[-1]-Hs[-2]))
    return math.log(lam)/L

if __name__ == "__main__":
    pats,_ = enumerate_patterns(1,1)
    tri = [c for c,comp,ty in pats if all(len(s)==6 for s in comp.values())][0]
    # fine grid in u = ln y, symmetric about 0
    us = [round(-0.6 + 0.1*k, 4) for k in range(13)]   # -0.6..0.6
    ys = [math.exp(u) for u in us]
    print("triangle lattice: susceptibility chi_L = d^2 f/du^2  (u=ln y)\n")
    for L in (2, 3):
        fs = [f_L(tri, 1, 1, L, y, Hmax=(12 if L<3 else 9)) for y in ys]
        print(f"L={L}:")
        peak = 0.0; peaku = None
        for k in range(1, len(us)-1):
            chi = (fs[k+1] - 2*fs[k] + fs[k-1])/((us[k+1]-us[k])**2)
            if abs(chi) > peak: peak, peaku = abs(chi), us[k]
            print(f"   u={us[k]:+.2f} (y={ys[k]:.3f})  f={fs[k]:.5f}  chi={chi:.4f}")
        print(f"   peak |chi| = {peak:.4f} at u={peaku}\n")
    print("interpretation: chi bounded & ~flat across L => NO transition (analytic free energy);")
    print("chi peak growing sharply with L => folding phase transition.")
