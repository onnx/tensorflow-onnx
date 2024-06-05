#!/bin/bash

# # Check if the argument is provided
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <tensorflow_version>"
#     exit 1
# fi

# Assign the argument to a variable
TF_VERSION=$1

pip install pytest pytest-cov pytest-runner coverage graphviz requests pyyaml pillow pandas parameterized sympy coloredlogs flatbuffers timeout-decorator
pip install onnx
pip install onnxruntime==1.16.3
pip install numpy

pip install onnxruntime-extensions
pip install "tensorflow-text<=$TF_VERSION"

pip uninstall -y tensorflow
pip install tensorflow==$TF_VERSION
pip uninstall -y protobuf
pip install "protobuf~=3.20"

python setup.py install

pip freeze --all