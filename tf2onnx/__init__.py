# SPDX-License-Identifier: Apache-2.0

"""tf2onnx package."""

__all__ = ["utils", "graph_matcher", "graph", "graph_builder",
           "tfonnx", "shape_inference", "schemas", "tf_utils", "tf_loader", "convert"]

import importlib
import onnx
from .version import git_version, version as __version__
from . import verbose_logging as logging

# NOTE: Importing heavily submodules here leads to a RuntimeWarning
#    when launching "python -m tf2onnx.convert":
#   "'tf2onnx.convert' found in sys.modules after import of package 'tf2onnx',
#   but prior to execution of 'tf2onnx.convert'; this may result in unpredictable behaviour"
# To prevent that, lazily import submodules on attribute access using PEP 562.

def __getattr__(name):
    """Lazily import submodules listed in __all__ on attribute access."""
    if name in __all__:
        module = importlib.import_module(f"tf2onnx.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Expose package attributes and available submodules for tab-completion."""
    return sorted(list(globals().keys()) + __all__)
