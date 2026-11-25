"""Check the WDM catalogue against its native mass-space ODE."""

from unittest.mock import patch
import warnings

import numpy as np
import pytest

import sashimi_w as model
from picard_tidal_stripping import PicardFallbackWarning


def test_paired_host_conversion_against_scalar_roots():
    """Check virial mass conversion at paired WDM host masses and redshifts.

    Notes
    -----
    Independent scalar calls solve the original radius equation in CGS
    units. Requiring agreement within 1e-8 relative mass tests numerical
    root solving and the pairing of mass and epoch. Both paths use the
    same concentration prescription.
    """
    halo = model.subhalos()
    mass = np.array([1e10, 2e11, 7e11, 1e12]) * model.Msolar
    epochs = np.array([2.7, 1.3, 0.6, 0.0])
    expected = [np.asarray(halo.Mvir_from_M200(m, z)).item() for m, z in zip(mass, epochs)]
    np.testing.assert_allclose(
        halo._Mvir_from_M200_grid(mass, epochs), expected, rtol=1e-8, atol=0.0
    )


@pytest.mark.parametrize("order,offset", [(8, 0.0), (1, 0.5)])
def test_default_catalogue_against_native_ode(order, offset):
    """Compare bound masses for the median and an offset WDM host history.

    Parameters
    ----------
    order : int
        Hermite order for host histories. Order one selects a single offset.
    offset : float
        Dimensionless number of host-history standard deviations.

    Notes
    -----
    Each small catalogue is built twice on the same native WDM model.
    Picard must run without fallback, match tight LSODA endpoint masses
    within 0.1 percent, and preserve the accretion-mass and redshift axes.
    This finite comparison uses the model's own power spectrum and rates.
    """
    halo = model.subhalos()
    config = dict(
        M0=1e12,
        redshift=0.0,
        dz=0.5,
        zmax=1.5,
        N_ma=12,
        N_herm=3,
        N_hermNa=order,
        sigmafac=offset,
        logmamin=8.0,
        logmamax=10.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", PicardFallbackWarning)
        actual = halo.rs_rhos_calc(**config)
    expected = halo.rs_rhos_calc(**config, method="odeint", rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(actual[4], expected[4], rtol=1e-3, atol=0.0)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_observable_wrappers_forward_stripping_method():
    """Pass the selected stripping solver from each observable to the catalogue.

    Notes
    -----
    Stop at catalogue entry to inspect the method and solver kwargs without
    depending on binning or survival conventions. The default and the
    explicit legacy override must both propagate through all three wrappers.
    """
    halo = model.subhalos()
    for call, extra in [
        (halo.subhalo_distr, {}),
        (halo.N_sat, {}),
        (halo.N_sat_Vthres, {"Vpeak_max": 1.0}),
    ]:
        for options in [{}, {"method": "odeint", "rtol": 2e-9}]:
            with patch.object(halo, "rs_rhos_calc", side_effect=StopIteration) as catalogue:
                with pytest.raises(StopIteration):
                    call(1e12, **extra, **options)
            forwarded = catalogue.call_args.kwargs
            assert forwarded["method"] == options.get("method", "picard_table")
            if "rtol" in options:
                assert forwarded["rtol"] == options["rtol"]
