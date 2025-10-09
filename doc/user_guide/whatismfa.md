# What is multifractal analysis?

The aim of multifractal analysis is to characterize all scaling properties of *scale invariant* objects (function, signal, image, measure, etc.).

:::{admonition} Scale invariance
:class: note
A **scale invariant** object does not possess a characteristic scale: its properties do not depend on the scale at which it is considered.
:::

Applying a scaling transformation (dilation or contraction) to a scale invariant object, the resulting object is similar to the starting object up to an appropriate multiplicative factor.
Alternatively, this means that upon being given a representation of the object, it is not possible to determine at which scale this representation was obtained purely based on its observation.

An object is strictly scale invariant if rescaling by a certain factor yields the exact same shape. The Koch snowflake is an example of strict scale invariance, as is the roman cauliflower.

In a broad sense, such as for random processes, scale invariance means that the properties of an object are preserved across scales, and not its exact form. For instance, the patterns of financial time series are not exactly reproduced across multiple temporal scales, yet their variance on windowed segments is typically scale invariant as a function of window length.

**Multifractal analysis** allows us to fully determine the theoretical scaling properties of scale invariant mathematical objects, and provides us with estimators of those properties on finite data sets.
The pymultifracs toolbox aims to provide an easy-to-use interface to estimate the multifractal properties of sampled data.
