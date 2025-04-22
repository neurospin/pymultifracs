# pylint: disable=C0114

from ._version import __version__  # noqa: F401

from .mf_analysis import mfa  # noqa: F401
from .wavelet import wavelet_analysis  # noqa: F401

try:
    from .robust.benchmark import Benchmark  # noqa: F401
except ModuleNotFoundError:
    pass
