import sys
import importlib
import pytest
import numpy as np


@pytest.fixture
def run_analysis(mocker, mrw_file):

    fname = mrw_file[0]

    with open(fname, 'rb') as f:
        X = np.load(f)

    output = {}

    import pymultifracs.backend
    import pymultifracs

    WTpL = pymultifracs.wavelet_analysis(X).integrate(1).get_leaders(2)

    scaling_ranges = [(3, WTpL.j2_eff())]

    pwt = pymultifracs.mfa(WTpL, scaling_ranges)

    output['jax'] = [
        WTpL.values, pwt.cumulants, pwt.structure
    ]

    mocker.patch.dict(
        sys.modules,
        {'jax': None,
            'jax.numpy': None}
    )

    importlib.reload(pymultifracs.backend)
    importlib.reload(pymultifracs)

    WTpL = pymultifracs.wavelet_analysis(X).integrate(1).get_leaders(2)
    pwt = pymultifracs.mfa(WTpL, scaling_ranges)

    output['no jax'] = [
        WTpL.values, pwt.cumulants, pwt.structure
    ]

    yield output

    importlib.reload(pymultifracs.backend)
    importlib.reload(pymultifracs)


@pytest.mark.jax
def test_numpy_wavelet(run_analysis):

    WTpL_val_jax = run_analysis['jax'][0]
    WTpL_val_nojax = run_analysis['no jax'][0]

    cumulants_jax = run_analysis['jax'][1]
    cumulants_nojax = run_analysis['no jax'][1]

    structure_jax = run_analysis['jax'][2]
    structure_nojax = run_analysis['no jax'][2]

    # Check that they have the same number of output scales
    assert max(WTpL_val_jax) == max(WTpL_val_nojax)

    for scale in WTpL_val_jax:
        np.testing.assert_allclose(
            WTpL_val_jax[scale], WTpL_val_nojax[scale], rtol=1e-6)

    np.testing.assert_allclose(
        cumulants_jax.values, cumulants_nojax.values, rtol=1e-6)

    np.testing.assert_allclose(
        structure_jax.values, structure_nojax.values, rtol=1e-6)
