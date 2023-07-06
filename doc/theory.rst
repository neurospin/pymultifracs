======
Theory
======

Here are presented some of the theoretical aspects required to understand multifractal analysis

Scale Invariance
----------------

A process :math:`X` is said scale invariant if its two-point function :math:`\mathcal{G}_{\alpha}(r)` scales as:

.. math:: \forall b \in \mathbb{R}_+^*, \quad \mathcal{G}_{\alpha}(r/b) = b^{2x_{\alpha}}\mathcal{G}(r)

In other words, the rescaling factor of the two-point function only depends on the scaling factor :math:`b`, and does not depend on the characteristic scale :math:`r`. As a consequence, by choosing :math:`b=r`, we can rewrite this relation as a power law:

.. math:: \mathcal{G}_{\alpha}(r) = \frac{\mathrm{constant}}{r^{2x_{\alpha}}}

Self-similarity
```````````````

For a 1D time series, we can choose the autocorrelation function as the two-point function:

.. math:: A(\tau) =  

Taking the Fourier transform, we have the following relation on the power spectrum:

.. math:: \Gamma(f) = 

Thus we have a power spectrun with a power law relation, appearing linear in a log-log plot. The exponent of that power law is related to the Hurst exponent for self-similarity, with the following relation:

The power spectrum investigate the scaling of the second-order moments of the time series increments:

Multifractality
```````````````

We can investigate the power law scaling of any arbitrary moment :math:`q` of :math:`\Delta X`, described using the scaling function :math:`\zeta(q)`:

.. math:: \mathbb{E}\left[ \Delta X ^q \right] \propto 

Time series where :math:`\zeta(q)=H`, i.e. there is only one exponent :math:`H` common to every moment, are called monofractal.

In the case where :math:`\zeta(q)` is not constant, then the time series is described as multifractal.

Regularity
----------

For a function, the notion of regularity captures the notion of .
There are multiple characterisations of regularity, such as Lipschitz, etc. Here we will focus on those relevant for multifractal analysis

Hölder regularity
`````````````````


:math:`p`-exponent
``````````````````


Multifractal formalism
----------------------

With discrete measurements, in order to characterize :math:`\zeta(q)` we cannot directly measure the pointwise regularity of our time series.
Instead, we need to rely on *multi-resolution quantities* (MRQ) which describe the statistics of the underlying process. From those MRQ we can then derive estimates of the scaling function, and of the multifractal spectrum.

We note the MRQ as :math:`T(k, j)` where :math:`k` is a shift term, and :math:`j` is the scale.

Structure functions
```````````````````

In order to estimate :math:`\zeta(q)`, we can look at the structure functions :math:`S`, defined as:

.. math:: S(j, q) = 2^{dj} \sum_k T(k, j)^q

Then we have:

.. math:: \zeta(q) = \varliminf_{j \to -\infty} \frac{\log \left( S(j, q) \right)}{\log(2^j)}

Multifractal Spectrum
`````````````````````

From the structure functions we can get an upper-bound estimate of the multifractal spectrum via the Legendre transform of the scaling function.

.. math:: \mathcal{L}(h) = \inf_{q\in \mathbb{R}} \left(d + qh - \zeta(q) \right)

Cumulants
`````````

The wavelet formalism provides a convenient way to characterize the properties of the multifractal spectrum.
Using the cumulant expansion of :math:`\zeta(q)`:

.. math:: \zeta(q) = \sum_{m\geq 1} c_m \frac{q^m}{m!}

We can estimate the :math:`c_m` from the scaling of the cumulants of the log MRQ:

.. math:: \forall j, \quad \log \operatorname{Cumulant}_m \left[\left( T(k, j)\right)_k \right] \propto j c_m

Those :math:`c_m` provide a description of the moments of the spectrum:

- :math:`c_1` corresponds to the mode of the spectrum, and to self-similarity
- :math:`c_2` corresponds to twice the spread of the multifratal spectrum
- :math:`c_3` and :math:`c_4` describe asymmetry and tailedness respectively.

As it turns out, each definition of regularity has its corresponding multi-resolution quantity.

Wavelet coefficient
```````````````````

While the discrete wavelet transform is a useful tool for computing the Hurst exponent, it fails at properly capturing the regularity of time series

Wavelet leader
``````````````

The multi-resolution quantity associated with the Hölder exponent is the wavelet leader, defined as:

.. math::

Wavelet :math:`p`-leader
````````````````````````

Models for scale invariant signals
----------------------------------

fBM
```

MRW
```

Practical considerations
------------------------

Fractional integration
``````````````````````

Higher order cumulants
``````````````````````

Sensitivity to noise
````````````````````

Comparison with alternative methods
-----------------------------------

References
----------

H. Wendt, Contributions of Wavelet Leaders and Bootstrap to Multifractal Analysis, ENS Lyon, 2008. Available at https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf

