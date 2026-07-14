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

pip install "$NUMPY_SPEC" onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION onnxruntime-extensions
pip install pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator
pip install tensorflow==$TF_VERSION

pip install -e .

echo "----- List all of depdencies:"
pip freeze --all
