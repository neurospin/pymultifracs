# PyMultiFracs quickstart

Multifractal analysis in pymultifracs is split in two steps:
1. Computing multi-resolution quantities
2. Estimating the scaling functions (functions of $j$), and their associated scaling exponents:
    - Structure functions yield the scaling function $\zeta(q)$;
    - Cumulant scaling functions $C_m(j)$ yield the log-cumulants $c_m$;
    - The multifractal direct determination method yields pairs $(h(q), D(h(q)))$.

The scaling exponents are estimated by linear regression of the (log) scaling functions against $j$ on a given range of scales $[j_1, j_2]$.

## Multi-resolution quantity

pymultifracs relies on the {py:mod}`pywavelets` implementation of the discrete wavelet transform, and wraps it in the {class}`~pymultifracs.multiresquantity.WaveletDec` class.

An input array $\{x_t\}_{t \in [1, T]}$ is decomposed into wavelet coefficients $c_x(j, k)$, defined at scale $j$ and shift $k$.
The discrete wavelet transform decomposes the signal into scale-shift space dyadically: $j$ represents the (base 2) logarithm of the inverse normalized frequency. Accordingly, when $j$ increases by 1, the resolution of $k$ is halved.

<!-- For time series, $j$ is proportional to the logarithm of the inverse of the normalized frequency, and the $k$ indices index the temporal dimension, such that the shift corresponding to a time point $t$ may be written $k(t)$. -->

<!-- , which means that the number of coefficients at scale $j$ is proportional to $N$ -->
<!-- The scale $j=1$ corresponds to the Nyquist frequency, and hypothetically, the scale $j=0$ would correspond to the sampling frequency. -->

The exact correspondance between $j$ and frequency depends on the wavelet family that is chosen. In pymultifracs, frequency may be converted to scale and vice versa using the methods {meth}`~pymultifracs.multiresquantity.WaveletDec.freq2scale` and {meth}`~pymultifracs.multiresquantity.WaveletDec.scale2freq`:

```python
WT = wavelet_analysis(X)
# To which scale corresponds the frequency f? Returns a floating number
WT.freq2scale(f, sfreq)
# Which frequency corresponds to scale j=4?
WT.scale2freq(j, sfreq)
```

### Wavelet ($p$-)leaders

Wavelet coefficients are a natural choice for estimating scaling exponents, due to their dyadic decomposition of the data.
However, wavelet coefficients are only able to be used for estimating the structure functions for $q \geq 0$, for instance $q=2$ which yields an estimator of the Hurst exponent. $q < 0$ leads to unstable estimation using the wavelet coefficients.
<!-- However, since wavelet coefficients may take values arbitrarily close to zero, the statistics of their moments for $q < 0$ are not stable. -->

For multifractal analysis, it is important to use a well-behaved multi-resolution quantity, whose choice will determine the **multifractal formalism** upon which the analysis will be based. The wavelet leaders and $p$-leaders, derived from the wavelet coefficicents, allow the estimation of the multifractal spectrum and log-cumulants.

In pymultifracs, wavelet $p$-leaders are simply obtained by using the {meth}`~pymultifracs.multiresquantity.WaveletDec.get_leaders` method.

```python
WTpL = WT.get_leaders(p_exp=2)
```

Formally, wavelet leaders correspond to wavelet $p$-leaders when $p\to +\infty$, so they are computed in pymultifracs by passing `p_exp=np.inf`

```python
WTL = WT.get_leaders(p_exp=np.inf)
```

## Multifractal analysis

## The ``xarray.DataArray`` structure

The output of multifractal analysis in pymultifracs are instances of the {class}`xarray.DataArray` class.

The ``DataArray`` structure is a wrapper around numpy's ndarrays, that labels the dimensions, and may additionally provide coordinates for those dimensions. Internally, this is done to make sure that dimensions are properly aligned when applying operations.
This also means that the output arrays are immediatly interpretable, since the meaning of each dimension, and where relevant, the coordinates of the dimensions are provided along with the input.

``xarray`` can be thought of as extending the pandas dataframe approach, which is limited to 2D arrays, to a N-dimensional setting.

### Indexing data arrays

DataArrays provide a simple interface to select a particular entry by specifying the dimension by name. For instance, if ``pwt`` is the output of {func}`~pymultifracs.mfa`, the value of $C_2(3)$ can be obtained with the {meth}`~xarray.DataArray.sel` method:

```python
pwt.cumulants.values.sel(j=3, m=2)
```

To instead take $C_2(j)$ for j between $2$ and $8$, slicing with `np.s_` may be used:

```python
pwt.cumulants.values.sel(j=np.s_[2:8], m=2)
```

On dimensions where there are no coordinates, the {meth}`~xarra.DataArray.isel` method may be used to select by position.
For instance, if a 2D array was provided as input to {func}`~pymultifracs.wavelet_analysis`, then the $c_2$ estimate for the second entry can be obtained by:

```python
pwt.cumulants.c2.isel(channel=1)
```

For further details, please refer to the [xarray documentation](https://docs.xarray.dev/en/latest/user-guide/indexing.html)

### Quick plotting

On top of the more detailed plotting functions provided by the pymultifracs package, the contents of xarrays may be plotted using their {meth}`~xarray.DataArray.plot` method.

As an example, plotting the average structure function $S_2(j)$ across all channels and $j$ in the range $[2, 8]$ is as simple as:

```python
pwt.structure.values.sel(q=2, j=np.s_[2:8]).mean(dim='channel').plot()
```
