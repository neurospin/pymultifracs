from dataclasses import dataclass


@dataclass
class MultiResolutionQuantity:
    """
    The goal of this class is to provide methods to easily handle with the
    multiresolution quantities used in multifractal analysis.

    It can be used to represent wavelet coefficients d_X(j, k)
    and wavelet leaders L_X(j, k).

    Args:
        name (str)   : 'wavelet_coeffs' or 'wavelet_leaders'
        nj  (dict)   : nj[j] contains the number of coefficients at the scale j
        values (dict): values[j] contains the list of coefficients
                       at the scale j
    """
    name: str = 'wcmf'
    nj: dict = None
    values: dict = None

    def __post_init__(self):

        self.nj = {} if self.nj is None else self.nj
        self.values = {} if self.values is None else self.values

        self.name = {'wcmf': 'wavelet_coeffs',
                     'wlmf': 'wavelet_leaders',
                     'p-leader': 'wavelet_leaders'}[self.name]

    def add_values(self, coeffs, j):
        self.values[j] = coeffs
        self.nj[j] = len(coeffs)

    def get_nj(self):
        """
        Returns nj as a list
        """
        return list(self.nj.values())

    def get_nj_interv(self, j1, j2):
        """
        Returns nj as a list, for j in [j1,j2]
        """
        nj = []
        for j in range(j1, j2+1):
            nj.append(self.nj[j])
        return nj
