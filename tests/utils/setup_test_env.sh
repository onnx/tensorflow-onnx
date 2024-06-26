#!/bin/bash

# # Check if the argument is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <tensorflow_version> <onnxruntime_version> <onnx_version>"
    exit 1
fi

# Assign the argument to a variable
TF_VERSION=$1
ORT_VERSION=$2
ONNX_VERSION=$3

echo "==== TensorFlow version: $TF_VERSION"
echo "==== ONNXRuntime version: $ORT_VERSION"
echo "==== ONNX version: $ONNX_VERSION"

pip install pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator
pip install onnx==$ONNX_VERSION
pip install onnxruntime==$ORT_VERSION
pip install "numpy<2"

pip install onnxruntime-extensions
pip install "tensorflow-text<=$TF_VERSION"

pip uninstall -y tensorflow
pip install tensorflow==$TF_VERSION
pip uninstall -y protobuf
pip install "protobuf~=3.20"

python setup.py install

echo "----- List all of depdencies:"
pip freeze --all