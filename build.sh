#!/bin/bash

set -x

apt-get install -y protobuf-compiler libprotoc-dev
pip install setuptools 
pip install onnx pytest-cov

python setup.py test
python setup.py bdist_wheel
