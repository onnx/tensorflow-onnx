#!/bin/bash

# Prototype: same interface as setup_test_env.sh, but resolves every
# dependency - including the editable tf2onnx install - with uv in a single
# pass instead of pip's sequence of independent installs. A single resolve
# sees NUMPY_SPEC alongside every transitive requirement at once, so a
# transitive requirement that's incompatible with it becomes a resolution
# error instead of silently overriding it (the class of bug behind the
# ml_dtypes 0.6.0 incident this lane exists to guard against).

# Fail fast: stop on the first error (e.g. a failed uv install) and on unset
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

uv pip install --system \
    "$NUMPY_SPEC" \
    onnx==$ONNX_VERSION onnxruntime==$ORT_VERSION onnxruntime-extensions \
    tensorflow==$TF_VERSION \
    pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator \
    -e .

echo "----- List all of dependencies:"
uv pip list --system
