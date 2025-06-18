# Release notes

### v0.3.1

API changes:

New features:
- Switched from numpy arrays to xarray DataArrays.
- Documentation example for bootstrapping.

Bug fixes:
- Fixed bootstrapping for bivariate analysis.

Other updates:
- Switch to proper versioning of the github repository.
- Tests for bootstrapping.
- Cleaned up legacy code and notebooks.

## v0.3

API changes:
- Rework of the API for working with multi-resolution quantities, for simpler use.
- Updated Benchmarking code.

New features:
- New documentation style: from notebook to python scripts.
- Outlier detection via the `get_outliers` function.
- Robust cumulants for multifractal analysis.
- Documented analysis of simulations, bivariate analysis and benchmarking.
- Weak-scaling exponents are available as MRQ.

Other updates:
- Improved test coverage across the board.
- Switched to github actions for CI
