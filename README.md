<p align="center">
  <img src="assets/logo.svg" alt="SASHIMI-W logo" width="440">
</p>

# Semi-Analytical SubHalo Inference ModelIng for WDM (SASHIMI-W)
[![arXiv](https://img.shields.io/badge/arXiv-2111.13137%20-green.svg)](https://arxiv.org/abs/2111.13137)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for wold dark matter (WDM). The results are well in agreement with those from numerical N-body simulations.


## Tidal mass loss

The WDM catalogue and its distribution and satellite-count methods use
`method="picard_table"` to interpolate bound mass at the observation
redshift. Pass `method="odeint"` to select the original mass-space ODE solver.
Picard integrates the native stripping equation
from accretion to observation using three nonlinear updates. A fourth update
checks iteration convergence. Quadrature and interpolation also contribute to
the mass error. The table covers `0 <= z <= za <= 7` and
`-24 <= log10(ma/Mvir(za)) <= 3`. Outside these limits or when the iteration
check fails, the solver issues a warning and directly integrates log mass.

## References

If you use the SASHIMI script to write a paper, please cite:

- A. Dekker, S. Ando, C.A. Correa and K.C.Y. Ng, arXiv: 2111.13137
- N. Hiroshima, S. Ando and T. Ishiyama, arXiv: 1803.07691
- A.D. Ludlow, S. Bose, R.E. Angulo, L. Wang, W.A. Hellwing, J.F. Navarro, S. Cole and C.S. Frenk, arXiv: 1601.02624
