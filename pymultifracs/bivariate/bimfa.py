from collections import namedtuple

from ..mfa import MFractalData
from .bivariate_cumulants import BiCumulants
from .bivariate_structurefunction import BiStructureFunction

MFractalBiVar = namedtuple('MFractalBiVar', 'structure cumulants')


def bivariate_analysis(WT1, WT2, j1, weighted, n_cumul, q1, q2):

    j2 = min(WT1.j2_eff, WT2.j2_eff)

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

    dwt = MFractalBiVar(dwt_bicm, dwt_bisf)

    if WT1.wt_leaders is not None and WT2.wt_leaders is not None:

        param_lwt = {'mrq1': WT1.wt_leaders, 'mrq2': WT2.wt_leaders,
                     **parameters}

        lwt_bicm = BiCumulants.from_dict(param_lwt)
        lwt_bisf = BiStructureFunction.from_dict(param_lwt)

        lwt = MFractalBiVar(lwt_bicm, lwt_bisf)

    return MFractalData(dwt, lwt)
