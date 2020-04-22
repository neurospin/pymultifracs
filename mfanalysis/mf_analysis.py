from collections import namedtuple

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis
from .estimation import estimate_hmin

MFractalData = namedtuple('MFractalData', 'dwt lwt')
MFractalVar = namedtuple('MFractalVar', 'structure cumulants spectrum')


def mf_analysis(wt_coefs, wt_leaders, j2_eff, p_exp, j1, weighted,
                n_cumul, q):
    """
    Perform multifractal analysis

    Parameters
    ----------
    """

    estimate_hmin(wt_coefs, j1, j2_eff, weighted)

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2_eff,
        'wtype': weighted,
    }

    param_dwt = {
        'mrq': wt_coefs,
        **parameters
    }

    dwt_struct = StructureFunction(**param_dwt)
    dwt_cumul = Cumulants(**param_dwt)
    dwt_spec = MultifractalSpectrum(**param_dwt)

    dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec)

    if p_exp is not None:

        param_lwt = {
            'mrq': wt_leaders,
            **parameters
        }

        lwt_struct = StructureFunction(**param_lwt)
        lwt_cumul = Cumulants(**param_lwt)
        lwt_spec = MultifractalSpectrum(**param_lwt)

        lwt = MFractalVar(lwt_struct, lwt_cumul, lwt_spec)

    else:

        lwt = None

    return MFractalData(dwt, lwt)


def mf_analysis_full(signal, j1=1, j2=10, normalization=1, gamint=0.0,
                     weighted=True, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3):

    if q is None:
        q = [2]

    wt_transform = wavelet_analysis(signal, p_exp=p_exp, wt_name=wt_name,
                                    j1=j1, j2=j2, gamint=gamint,
                                    normalization=normalization,
                                    weighted=weighted)

    mf_data = mf_analysis(wt_transform.wt_coefs,
                          wt_transform.wt_leaders,
                          wt_transform.j2_eff,
                          p_exp=p_exp,
                          j1=j1,
                          weighted=weighted,
                          n_cumul=n_cumul,
                          q=q)

    return mf_data
