from collections import namedtuple

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis, _check_formalism

MFractalData = namedtuple('MFractalData', 'structure cumulants spectrum')


def mf_analysis(wt_coefs, wt_leaders, j2_eff, p_exp, j1, weighted,
                n_cumul, q):
    """
    Perform multifractal analysis

    Parameters
    ----------


    """

    formalism = _check_formalism(p_exp)

    # h_min, _ = estimate_hmin(wt_coefs, j1, j2_eff, weighted)

    parameters = {
        'mrq': wt_coefs if formalism == 'wcmf' else wt_leaders,
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2_eff,
        'wtype': weighted,
    }

    structure = StructureFunction(**parameters)
    cumulants = Cumulants(**parameters)
    spectrum = MultifractalSpectrum(**parameters)

    return MFractalData(structure, cumulants, spectrum)


def mf_analysis_full(signal, j1=1, j2=10, normalization=1, gamint=0.0,
                     weighted=True, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3):

    if q is None:
        q = [2]

    wt_transform = wavelet_analysis(signal, p_exp=p_exp, wt_name=wt_name,
                                    j1=j1, j2=j2, gamint=gamint,
                                    normalization=normalization,
                                    weighted=weighted)

    structure, cumulants, spectrum = mf_analysis(wt_transform.wt_coefs,
                                                 wt_transform.wt_leaders,
                                                 wt_transform.j2_eff,
                                                 p_exp=p_exp,
                                                 j1=j1,
                                                 weighted=weighted,
                                                 n_cumul=n_cumul,
                                                 q=q)

    return structure, cumulants, spectrum
