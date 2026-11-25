"""Compare the Picard and odeint masses in the WDM catalogue."""

import warnings

import numpy as np

import sashimi_w as model
from picard_tidal_stripping import PicardFallbackWarning


def test_picard_endpoints_against_odeint():
    """Compare the predicted bound masses at z=0 for one host.

    Notes
    -----
    Both solvers use a 1e12 solar-mass host with three accretion epochs
    and twelve masses. A Picard fallback or a relative difference above
    0.1 percent from tight mass-space odeint fails the comparison.
    This checks numerical agreement for the sampled host and grid.
    """
    catalogue = model.subhalos().rs_rhos_calc
    grid = dict(M0=1e12, dz=0.5, zmax=1.5, N_ma=12, N_herm=3, N_hermNa=8,
                logmamin=8.0, logmamax=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PicardFallbackWarning)
        picard = catalogue(**grid, method="picard_table")[4]
    odeint = catalogue(**grid, method="odeint", rtol=1e-10, atol=1e-11)[4]
    np.testing.assert_allclose(picard, odeint, rtol=1e-3, atol=0.0, equal_nan=False)
