#!/bin/bash

# Fail fast: stop on the first error (e.g. a failed pip install) and on unset
# variables, so a version-matrix job never silently proceeds with a broken env.
set -euo pipefail

# # Check if the argument is provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <tensorflow_version> <onnxruntime_version> <onnx_version> [numpy_spec]"
    exit 1
fi

# Assign the argument to a variable
TF_VERSION=$1
ORT_VERSION=$2
ONNX_VERSION=$3
# numpy constraint is configurable so a lane can exercise the suite under numpy
# 2.x; default keeps the historical numpy<2 pin for the existing combinations.
NUMPY_SPEC="${4:-numpy<2}"

echo "==== TensorFlow version: $TF_VERSION"
echo "==== ONNXRuntime version: $ORT_VERSION"
echo "==== ONNX version: $ONNX_VERSION"
echo "==== numpy spec: $NUMPY_SPEC"

# Extra insurance only: onnx pulls in ml_dtypes via "ml_dtypes>=0.5.0", and
# ml_dtypes 0.6.0 hard-requires numpy>=2.0. This first resolution already
# honours NUMPY_SPEC on its own; the pin just keeps ml_dtypes numpy<2-compatible.
pip install "$NUMPY_SPEC" "ml_dtypes<0.6.0" onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION onnxruntime-extensions
# Re-assert NUMPY_SPEC on every subsequent install; this is the actual fix.
# Each `pip install` is an independent resolution: installing tensorflow
# downgrades ml_dtypes (e.g. to 0.2.0), and the final `pip install -e .` then
# re-resolves onnx's "ml_dtypes>=0.5.0" to the latest ml_dtypes (0.6.0), which
# drags numpy to 2.x unless NUMPY_SPEC is passed again.
pip install "$NUMPY_SPEC" pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator
pip install "$NUMPY_SPEC" tensorflow==$TF_VERSION

pip install "$NUMPY_SPEC" -e .

echo "----- List all of depdencies:"
pip freeze --all
