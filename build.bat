rem SPDX-License-Identifier: Apache-2.0

python -m pytest --cov=tf2onnx
python setup.py bdist_wheel
