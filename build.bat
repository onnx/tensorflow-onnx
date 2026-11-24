rem SPDX-License-Identifier: Apache-2.0

python -m pip install build
python -m pytest --cov=tf2onnx
python -m build
