#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python -m pip install build twine
python -m build
python -m twine check dist/*
