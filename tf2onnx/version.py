# SPDX-License-Identifier: Apache-2.0

"""Package version compatibility attributes.

``version`` and ``git_version`` are retained for callers that imported the
legacy generated module. New code should use ``tf2onnx.__version__``.
"""

import importlib.metadata

try:
    version = importlib.metadata.version("tf2onnx")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"

# Builds are now reproducible from VERSION_NUMBER and no longer modify the
# source tree to embed the current checkout's commit.
git_version = "unknown"
