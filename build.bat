@echo off
rem SPDX-License-Identifier: Apache-2.0

python -m pip install build twine || exit /b 1
python -m build || exit /b 1
python -m twine check dist\* || exit /b 1
