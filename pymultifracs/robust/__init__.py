# pylint: disable=C0114

try:
    from .robust import get_outliers  # noqa F401
    from .robust import compute_robust_cumulants  # noqa F401
except ModuleNotFoundError:
    pass
