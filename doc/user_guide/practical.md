# Practical considerations

## Choice of wavelet family

The wavelet decomposition function {func}`~pymultifracs.wavelet_analysis` allows chosing the wavelet family, by passing the `wt_name` argument.
The naming convention follows the [pywavelets](https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html) API, for which the wavelet name is composed of the wavelet family, followed by the number of vanishing moments.

For example, using a Daubechies wavelet with 2 vanishing moments:
```python
WT = wavelet_analysis(X, wt_name='db2')
```

From a theoretical perspective, and with data of infinite size, the choice of wavelet basis should not affect the results at all, given that there are enough vanishing moments.
Comparing the results obtained for an increasing number of vanishing moments should then ideally show minimal variation in the final exponents.

However, increasing the number of vanishing moment also reduces the number of valid coefficients available for the analysis, as it increases the length of the filters and thus increases the number of invalid wavelet coefficients at the boundaries.
For practical applications where the data is of finite size, this means that a compromise might be required in order to keep reasonably stable estimates or to preserve the upper temporal scales.

The default value is the Daubechies wavelet with 3 vanishing moments, `wt_name='db3'`, which should be a reasonable starting point in most cases.

## Regularity conditions and fractional integration

In order for multifractal analysis to be meaningful, some regularity conditions of the signal need to be respected. Those conditions depend on the multifractal formalism, but pymultifracs provides a unified method {meth}`~pymultifracs.multiresquantity.WaveletDec.check_regularity`, which will, given a scaling range, verify that the multi-resolution quantity are regular enough.

```python
WTpL.check_regularity([(3, 8)])
WTL.check_regularity([(3, 8)])
```
This is usually done automatically by {func}`pymultifracs.mfa`.

If the regularity criterion is not met, an error will be raised. In most cases, this can be addressed by fractionally integrating by a factor `gamint`, which is done via {meth}`~pymultifracs.multiresolutionquantity.WaveletDec.integrate`, common to all formalisms.

```python
WTpL = WT.integrate(gamint=1).get_leaders(2)
# equivalently,
WTpL = WT.get_leaders(2).integrate(gamint=1)
```
Typically, integrating by a factor gamint means that the multifractal spectrum will be shifted by `gamint` to higher values of $h$, and correspondingly $c_1$ will increase by `gamint` as well

:::{attention}
Integrating the multi-resolution quantity may affect the estimates beyond a simple shift of the multifractal spectrum, depending on the nature of the data. Before using a high value of gamint, consider other options: remove outliers, switch from wavelet leaders to wavelet $p$-leaders, reduce the value of `p_exp` while ensuring estimates are stable.
:::
