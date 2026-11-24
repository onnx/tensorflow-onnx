#!/bin/bash

# SPDX-License-Identifier: Apache-2.0


set -eux

apt-get install -y protobuf-compiler libprotoc-dev
python -m pip install build onnx pytest-cov

python -m pytest --cov=tf2onnx
python -m build
