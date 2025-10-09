# Basics of scale invariance analysis

Let us first clarify what is scale invariance

## Scale invariance

Consider the Koch fractal:

This notion of invariance to scaling operations can be extended to a larger class of functions, where the preserved property is not the exact shape but rather a quantity that is derived from the function over different scales.

The Brownian motion is an example of scale invariant random process, where the statistical properties of the process do not depend on the scale at which it is considered:

More formally, the notion of scale invariance is connected to the idea of **power law**, which describes scaling relationships $f$ along a variable $x$ of the following nature:
:::{math}
f(x) \propto x^{\alpha} \, .
:::

Where $\alpha$ is the **scaling exponent** of the power-law. We can see that it exhibits scale invariance, as rescaling by a particular lengthscale $b$ preserves the nature of the relationship and the power-law exponent:
```{math}
f(x/b) \propto b^{-\alpha} x^{\alpha} \propto x^{\alpha} \, .
```

### Hurst exponent

In general, for a scale invariant random process $X(t)$ we can define the **Hurst exponent** $H\in(0, 1)$, which characterizes the behavior of the process under rescaling:
$$
X(ct) = c^H X(t) \,.
$$

Brownian motion has a Hurst exponent $H=0.5$; there is a natural extension of Brownian motion that has an adjustable Hurst exponent: the **fractional Brownian motion** (fBM) model. An fBm $Y(t)$ with Hurst exponent $H$ is defined by its autocorrelation function:

$$
\mathbb{E}\left[ Y(t) Y(s) \right] = \frac{1}{2} \left( |t|^{2H} + |s|^{2H} - |t - s|^{2H} \right)\,.
$$

## Second-order analysis

