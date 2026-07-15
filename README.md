# Code for "Residual entropy of the degree-6 triangular origami lattice: a charge-±2 ice model"

Krishna Harish, Elkins High School. Reproduces every number in the paper.

## Requirements
Python 3.8+ and NumPy. (SymPy optional, only for `charpoly.py`.)

## One-command reproduction
```
python3 reproduce.py
```
Prints, next to the paper values: the transfer-matrix eigenvalues Λ_L and the residual
entropy bracket 3.7479350 ≤ μ ≤ 3.7479355 (s = ln μ = 1.321205); the exact
charged-ice = flat-fold match on the 2×2, 3×2, 2×3, 3×3 tori (402, 5594, 5594, 291438);
the new integer sequences; and the degree-4 baseline spectral match to the square-lattice
three-coloring model.

## What each file does (claim → script)
| Paper claim | Script |
|---|---|
| Single-vertex flat-fold engine (Kawasaki/Maekawa) | `vertex.py` |
| Global MV counting by factor graph / bucket elimination | `cp.py` |
| Crease patterns, per-vertex tables, torus counts | `pattern_atlas.py` |
| Exact row transfer matrix M_L, eigenvalues Λ_L | `general_tm.py` |
| Independent validation of M_L against torus counts | `torus_truth.py` |
| Exact integer characteristic polynomials of Λ_L | `charpoly.py` |
| Residual entropy bracket + no phase transition | `transition.py`, `reproduce.py` |
| Degree-4 baseline: bird's-foot = square 3-coloring spectrum | `universality.py`, `reproduce.py` |
| Degree-4 baseline: quad-mesh face graph = square lattice | `facegraph.py` |
| Degree-4 baseline: exact 8×4 integer identity (3,500,970) | `facematch.py` |

## Notes
- `M_L` has nonnegative integer entries; its Perron eigenvalue Λ_L is an exact algebraic
  number (largest root of the integer characteristic polynomial). Floating-point
  eigenvalues are used only to evaluate that root numerically.
- The charged-ice enumeration in `reproduce.py` is brute force over ±1 edge variables and
  is intended for the small tori that validate the transfer matrix.
