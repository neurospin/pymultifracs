from collections import namedtuple

from ..mfa import MFractalData
from ..wavelet import wavelet_analysis
from .bivariate_cumulants import BiCumulants
from .bivariate_structurefunction import BiStructureFunction

MFractalBiVar = namedtuple('MFractalBiVar', 'structure cumulants')


def bivariate_analysis(WT1, WT2, j1, weighted, n_cumul, q1, q2):

    j2 = min(WT1.wt_coefs.j2_eff(), WT2.wt_coefs.j2_eff())

    parameters = {
        'q1': q1,
        'q2': q2,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'n_cumul': n_cumul
    }

    param_dwt = {'mrq1': WT1.wt_coefs, 'mrq2': WT2.wt_coefs, **parameters}

    dwt_bicm = BiCumulants.from_dict(param_dwt)
    dwt_bisf = BiStructureFunction.from_dict(param_dwt)

    dwt = MFractalBiVar(dwt_bisf, dwt_bicm)

    if WT1.wt_leaders is not None and WT2.wt_leaders is not None:

        param_lwt = {'mrq1': WT1.wt_leaders, 'mrq2': WT2.wt_leaders,
                     **parameters}

        lwt_bicm = BiCumulants.from_dict(param_lwt)
        lwt_bisf = BiStructureFunction.from_dict(param_lwt)

        lwt = MFractalBiVar(lwt_bisf, lwt_bicm)

    return MFractalData(dwt, lwt)


def bivariate_analysis_full(signal1, signal2, j1, j2, normalization=1,
                            gamint=0.0, weighted=None, wt_name='db3',
                            p_exp=None, q1=None, q2=None, n_cumul=3):

    wt_param = {
        'p_exp': p_exp,
        'wt_name': wt_name,
        'j1': j1,
        'j2': j2,
        'gamint': gamint,
        'normalization': normalization,
        'weighted': weighted
    }

    WT1 = wavelet_analysis(signal1, **wt_param)
    WT2 = wavelet_analysis(signal2, **wt_param)

    return bivariate_analysis(WT1, WT2, j1, weighted, n_cumul, q1, q2)
