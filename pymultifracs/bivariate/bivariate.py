def bivariate_analysis(self, signal_1, signal_2):
    """
    Bivariate multifractal analysis according to:
        https://www.irit.fr/~Herwig.Wendt/data/Wendt_EUSIPCO_talk_V01.pdf

    Signals are cropped to have the same length,
        equal to min(len(signal_1), len(signal_2))
    """

    # TODO adapt it to signal_1 and signal_2 being instances of the Signal class

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
