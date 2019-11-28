import warnings
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from .cumulants import *
from .structurefunction import *
from .multiresquantity import *
from .mfspectrum import *
from .bivariate_structurefunction import *
from .bivariate_cumulants import *
# TODO: remove star imports

"""
The code in this package is based on the Wavelet p-Leader and Bootstrap based
MultiFractal analysis (PLBMF) Matlab toolbox written by Herwig Wendt
(https://www.irit.fr/~Herwig.Wendt/software.html) and on the documents
provided in his website (his PhD thesis in particular, which can be found at
https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf).
"""

"""
# NOTES:

* Computation of wavelet coefficients:
    - I had to multiply the PyWavelet high-pass filter by -1, that is
        -1*np.array(wt.dec_hi), in order to obtain the same wavelet
        coefficients as the matlab toolbox. Why? In the end, this should not
        change the result, since we take their absolute values.
    - Why is the index of the first good value equal to filter_len-2
        instead of filter_len-1? If filter_len == 2, there is still one
        corrupted value at the beginning.

* Possible bug reports for MatLab toolbox:
    - when nwt is changed, we need to do a clear all, otherwise the filter "h"
        may not have the correct size/values
    - signal of high-pass filter
    - number of wavelet leaders at j=1 is not correct the in
        mf_obj.mrq.leader.nj
"""


class MFA:
    """
    Class used for multifractal analysis.

    Args:
        formalism(str)  : 'wcmf'      - wavelet coefficient multifractal
                          'wlmf'      - wavelet leader multifractal
                          'p-leader'  - p-leader multifractal

        p (float)       :  p-leaders exponent, should be a positive value or
                           numpy.inf
                           Important: if p is not None, the parameter
                           `formalism` is ignored and the formalism is set
                           according to the value of p:
                           - for p > 0,      formalism = 'p-leader'
                           - for p = np.inf, formalism = 'wlmf'

        wt_name         : wavelet name (according to the ones available in
                          pywavelets, see:
                          https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html)
                          For instance:
                          'db2' - Daubechies with 2 vanishing moments

        j1 (int)        : smallest scale analysis

        j2 (int)        : largest scale analysis

        j2_eff(int)     : "effective" value of j2.
                          it is the minimum between j2 and the maximum
                          possible scale for wavelet decomposition, which
                          depends on the length of the data.

        max_level(int)  : maximum scale of wavelet decomposition

        wtype (int)     : 0 for ordinary regression, 1 for weighted regression

        gamint(float)   : fractional integration parameter


        q  (numpy.array): numpy array containing the exponents for which
                          to compute the structure functions

        n_cumul(int)      : number of cumulants to compute

        normalization(int): type of wavelet coefficient normalization
                            (1 for L1, 2 for L2 etc)

        weighted (bool)   : True for weighted linear regression,
                            False for ordinary linear regression

        verbose(int)      : verbosity level

        hi (numpy.array)  : wavelet high-pass filter

        lo (numpy.array)  : wavelet low-pass filter

        filter_len(int)   : length of the filters

        plt               : pointer to matplotlib.pyplot for convenience

        wavelet_coeffs (MultiResolutionQuantity) : stores wavelet coefficients

        wavelet_leaders (MultiResolutionQuantity): stores wavelet leaders

        structure(StructureFunction): structure function (depends on the
                                          chosen formalism)

        cumulants(Cumulants)                     : cumulants C_p(j)

        spectrum (MultifractalSpectrum)          : multifractal spectrum object

        hmin         : uniform regularity exponent

        eta_p (float): value of the wavelet scaling function at exponent p,
                       used to correct the values of the cumulants
                       when p-leaders are used

        hurst_structure (numpy.array): structure function log2(S(j, 2))
                                       computed in function compute_hurst()
    """
    def __init__(self,
                 formalism='wlmf',
                 p=None,
                 wt_name='db3',
                 j1=1,
                 j2=10,
                 gamint=0.0,
                 q=None,
                 n_cumul=3,
                 verbose=1,
                 normalization=1,
                 wtype=1
                 ):
        self.formalism = formalism
        self.wt_name = wt_name
        self.j1 = int(j1)
        self.j2 = int(j2)
        self.wtype = wtype
        self.gamint = gamint
        self.verbose = verbose
        self.q = [2] if q is None else q
        self.normalization = normalization
        self.n_cumul = n_cumul
        self.max_level = 0
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None
        self.spectrum = None
        self.hmin = None
        self.eta_p = None
        self.p = p

        # Set parameters (verify formalism, initialize wavelet filters)
        # and verify if everything is ok
        self._set_and_verify_parameters()

        # Keep pointer to matplotlib.pyplot for convenience
        self.plt = plt

        # Constants
        self.STRUCTURE_FIG_LABEL = 'Structure Functions'
        self.SCALING_FIG_LABEL = 'Scaling Function'
        self.CUMUL_FIG_LABEL = 'Cumulants'
        self.HMIN_FIG_LABEL = 'hmin'

        # Other
        self.utils = Utils()  # used for linear regression

    def _set_and_verify_parameters(self):
        """
        Called at __init__() and before each analysis, to set the parameters or
        adapt changed parameters.
        """

        # Check formalism
        _check_formalism(self.p)

        # Initialize wavelet filters
        self._initialize_wavelet_filters()

        # Verify j1
        assert self.j1 >= 1, "j1 should be equal or greater than 1."


    def plot_cumulants(self, show=False):
        """
        Plot cumulants
        """
        assert self.cumulants is not None, u"mfa.analyze() should be called\
                                             before plotting"
        self.cumulants.plot(self.CUMUL_FIG_LABEL)

        if show:
            self.plt.show()

    def plot_structure(self, show=False):
        """
        Plot structure function
        """
        assert self.structure is not None, u"mfa.analyze() should be called\
                                             before plotting"
        self.structure.plot(self.STRUCTURE_FIG_LABEL, self.SCALING_FIG_LABEL)

        if show:
            self.plt.show()

    def plot_spectrum(self, show=False):
        """
        Plot multifractal spectrum
        """
        assert self.spectrum is not None, u"mfa.analyze() should be called\
                                            before plotting"
        self.spectrum.plot()

        if show:
            self.plt.show()

    def analyze(self, signal):
        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        # Compute hmin
        self._estimate_hmin()

        # p-leader correction
        if self.formalism == 'p-leader':
            self._estimate_eta_p()
            self._correct_leaders()
        else:
            self.eta_p = np.inf

        # Compute structure functions and cumulants
        if self.formalism == 'wcmf':
            multi_res_quantity = self.wavelet_coeffs
        elif self.formalism == 'wlmf' or self.formalism == 'p-leader':
            multi_res_quantity = self.wavelet_leaders

        self.structure = StructureFunction(multi_res_quantity,
                                           self.q,
                                           self.j1,
                                           self.j2_eff,
                                           self.wtype)

        self.cumulants = Cumulants(multi_res_quantity,
                                   self.n_cumul,
                                   self.j1,
                                   self.j2_eff,
                                   self.wtype)

        self.spectrum = MultifractalSpectrum(multi_res_quantity,
                                             self.q,
                                             self.j1,
                                             self.j2_eff,
                                             self.wtype)

        # Plot
        if self.verbose >= 2:
            self.structure.plot(self.STRUCTURE_FIG_LABEL,
                                self.SCALING_FIG_LABEL)
            self.cumulants.plot(self.CUMUL_FIG_LABEL)
            self.spectrum.plot()
            self.plt.show()

    def compute_hurst(self, signal):
        """
        Estimate the Hurst exponent using the wavelet structure function
            for q=2
        """

        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        structure_dwt = StructureFunction(self.wavelet_coeffs,
                                          np.array([2.0]),
                                          self.j1,
                                          self.j2_eff,
                                          self.wtype)
        self.structure = structure_dwt

        log2_Sj_2 = np.log2(structure_dwt.values[0, :])  # log2(S(j, 2))
        self.hurst_structure = log2_Sj_2

        hurst = structure_dwt.zeta[0]/2

        if self.verbose >= 2:
            structure_dwt.plot(self.STRUCTURE_FIG_LABEL,
                               self.SCALING_FIG_LABEL)
            self.plt.show()

        return hurst

    def bivariate_analysis(self, signal_1, signal_2):
        """
        Bivariate multifractal analysis according to:
            https://www.irit.fr/~Herwig.Wendt/data/Wendt_EUSIPCO_talk_V01.pdf

        Signals are cropped to have the same length,
            equal to min(len(signal_1), len(signal_2))
        """

        # Adjust lengths
        length = min(len(signal_1), len(signal_2))
        signal_1 = signal_1[:length]
        signal_2 = signal_2[:length]

        # Verify parameters
        self._set_and_verify_parameters()

        # Clear previously computed data
        self.wavelet_coeffs = None
        self.wavelet_leaders = None

        # Compute multiresolution quantities from signals 1 and 2
        self._wavelet_analysis(signal_1)
        wavelet_coeffs_1 = self.wavelet_coeffs
        wavelet_leaders_1 = self.wavelet_leaders

        self._wavelet_analysis(signal_2)
        wavelet_coeffs_2 = self.wavelet_coeffs
        wavelet_leaders_2 = self.wavelet_leaders

        # Choose quantities according to the formalism
        if self.formalism == 'wcmf':
            mrq_1 = wavelet_coeffs_1
            mrq_2 = wavelet_coeffs_2
        elif self.formalism == 'wlmf' or self.formalism == 'p-leader':
            mrq_1 = wavelet_leaders_1
            mrq_2 = wavelet_leaders_2

        # Compute structure functions
        self.bi_structure = BiStructureFunction(mrq_1,
                                                mrq_2,
                                                self.q,
                                                self.q,
                                                self.j1,
                                                self.j2_eff,
                                                self.wtype)

        # Compute cumulants
        self.bi_cumulants = BiCumulants(mrq_1,
                                        mrq_2,
                                        1,
                                        self.j1,
                                        self.j2_eff,
                                        self.wtype)

        warnings.warn("Bivariate cumulants (class BiCumulants) are only\
            implemented for C10(j), C01(j) and C11(j). All moments are already\
            computed (BiCumulants.moments), we need to implement only the\
            relation between multivariate cumulants and moments.")

        if self.verbose >= 2:
            self.bi_structure.plot()
            self.bi_cumulants.plot(self.CUMUL_FIG_LABEL)
            self.plt.show()

    def _test(self, signal):
        """
        Used for development purposes
        """
        self._set_and_verify_parameters()
        self.wavelet_coeffs = None
        self.wavelet_leaders = None
        self.structure = None
        self.cumulants = None

        # Compute wavelet coefficients and wavelet leaders
        self._wavelet_analysis(signal)

        self.spectrum = MultifractalSpectrum(self.wavelet_leaders,
                                             self.q,
                                             self.j1,
                                             self.j2_eff,
                                             self.wtype)
