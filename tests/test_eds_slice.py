import numpy as np

from surface_pricer.fitting.eds_slice import EDSSabrSlice


def test_eds_slice_all_parameter_legacy_regression():
    strikes = np.exp(np.array([-5.0, -3.0, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 3.0, 5.0]))
    slice_ = EDSSabrSlice(1.0, 1.0, 0.3, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1)

    expected = np.array(
        [
            0.6091582667982873,
            0.49795788303695343,
            0.29211345495635294,
            0.23528059301818677,
            0.21393540682282516,
            0.30031963850832466,
            0.4341000775000891,
            0.5508832741063504,
            0.6816307574949739,
            1.0665502561074922,
            1.2508562501309652,
        ]
    )

    np.testing.assert_allclose(slice_.get_implied_vol(strikes), expected, rtol=0.0, atol=1.0e-12)


def test_eds_slice_preserves_deep_otm_tiny_option_prices():
    args = (
        6564.086816243905,
        6564.086816243905,
        0.25,
        8.0 / 252.0,
        -0.5 / 0.3,
        0.15 / 0.3,
        -0.45 / 0.3,
        -0.20 / 0.3,
        -0.3 / 0.3,
        -0.1 / 0.3,
    )
    slice_ = EDSSabrSlice(*args)
    strikes = np.array([7551.866, 7880.208])

    expected = np.array([0.069540813904686, 0.069540813904686])

    np.testing.assert_allclose(slice_.get_implied_vol(strikes), expected, rtol=0.0, atol=1.0e-11)
