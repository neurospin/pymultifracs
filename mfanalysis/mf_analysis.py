from collections import namedtuple

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis
from .estimation import estimate_hmin

MFractalData = namedtuple('MFractalData', 'dwt lwt')
MFractalVar = namedtuple('MFractalVar', 'structure cumulants spectrum hmin')


def mf_analysis(wt_coefs, wt_leaders, j2_eff, p_exp, j1, weighted,
                n_cumul, q, stat_fun):
    """
    Perform multifractal analysis

    Parameters
    ----------
    """

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2_eff,
        'wtype': weighted,
        'stat_fun': stat_fun
    }

    param_dwt = {
        'mrq': wt_coefs,
        **parameters
    }

    dwt_struct = StructureFunction(**param_dwt)
    dwt_cumul = Cumulants(**param_dwt)
    dwt_spec = MultifractalSpectrum(**param_dwt)

    dwt_hmin, _ = estimate_hmin(wt_coefs, j1, j2_eff, weighted)

    dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec, dwt_hmin)

    if p_exp is not None:

        param_lwt = {
            'mrq': wt_leaders,
            **parameters
        }

        lwt_struct = StructureFunction(**param_lwt)
        lwt_cumul = Cumulants(**param_lwt)
        lwt_spec = MultifractalSpectrum(**param_lwt)

        lwt_hmin, _ = estimate_hmin(wt_leaders, j1, j2_eff, weighted)

        lwt = MFractalVar(lwt_struct, lwt_cumul, lwt_spec, lwt_hmin)

    else:

        lwt = None

    return MFractalData(dwt, lwt)


def minimal_mf_analysis(wt_coefs, wt_leaders, j2_eff, p_exp, j1, weighted,
                        n_cumul, q, stat_fun):

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2_eff,
        'wtype': weighted,
        'stat_fun': stat_fun
    }

    dwt_struct = StructureFunction(mrq=wt_coefs, **parameters)
    dwt = MFractalVar(dwt_struct, None, None, None)

    lwt_cumul = Cumulants(mrq=wt_leaders, **parameters)
    lwt = MFractalVar(None, lwt_cumul, None, None)

    return MFractalData(dwt, lwt)


def mf_analysis_full(signal, j1=1, j2=10, normalization=1, gamint=0.0,
                     weighted=True, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3, minimal=False, stat_fun='mean'):

    if q is None:
        q = [2]

    wt_transform = wavelet_analysis(signal, p_exp=p_exp, wt_name=wt_name,
                                    j1=j1, j2=j2, gamint=gamint,
                                    normalization=normalization,
                                    weighted=weighted)

    fun = minimal_mf_analysis if minimal else mf_analysis

    mf_data = fun(wt_transform.wt_coefs,
                  wt_transform.wt_leaders,
                  wt_transform.j2_eff,
                  p_exp=p_exp,
                  j1=j1,
                  weighted=weighted,
                  n_cumul=n_cumul,
                  q=q,
                  stat_fun=stat_fun)

    return mf_data
