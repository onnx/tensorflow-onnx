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

# Pin ml_dtypes explicitly: onnx pulls it in transitively via "ml_dtypes>=0.5.0",
# and its latest release (0.6.0) hard-requires numpy>=2.0, which silently
# overrides NUMPY_SPEC during resolution. ml_dtypes<0.6.0 still satisfies onnx's
# constraint and stays numpy<2-compatible.
pip install "$NUMPY_SPEC" "ml_dtypes<0.6.0" onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION onnxruntime-extensions
# Re-assert NUMPY_SPEC on every subsequent install: each `pip install` is an
# independent resolution, so an unpinned transitive numpy requirement in any
# later package (tensorflow, pandas, tf2onnx itself, ...) can silently
# upgrade numpy past what the first command settled on.
pip install "$NUMPY_SPEC" pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator
pip install "$NUMPY_SPEC" tensorflow==$TF_VERSION

pip install "$NUMPY_SPEC" -e .

echo "----- List all of depdencies:"
pip freeze --all
