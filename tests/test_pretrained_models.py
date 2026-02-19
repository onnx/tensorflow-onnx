# SPDX-License-Identifier: Apache-2.0

"""Pytest wrapper for pretrained model tests, enables JUnit XML output."""

import os

import pytest
from packaging.version import Version

from run_pretrained_models import Test, load_tests_from_yaml
from tf2onnx import tf_utils

_CONFIG = os.path.join(os.path.dirname(__file__), "run_pretrained_models.yaml")
_OPSET = int(os.environ.get("TF2ONNX_TEST_OPSET", "18"))
_BACKEND = os.environ.get("TF2ONNX_TEST_BACKEND", "onnxruntime")
_SKIP_TF = os.environ.get("TF2ONNX_SKIP_TF_TESTS", "FALSE").upper() == "TRUE"
_SKIP_TFLITE = os.environ.get("TF2ONNX_SKIP_TFLITE_TESTS", "FALSE").upper() == "TRUE"
_SKIP_TFJS = os.environ.get("TF2ONNX_SKIP_TFJS_TESTS", "TRUE").upper() == "TRUE"

_TESTS = load_tests_from_yaml(_CONFIG)
_TEST_IDS = sorted(_TESTS.keys())

Test.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pretrained-models")
Test.target = []


def _skip_reason(test):
    if test.disabled:
        return "disabled"
    if _SKIP_TFJS and test.model_type == "tfjs":
        return "tfjs tests skipped"
    if _SKIP_TFLITE and test.model_type == "tflite":
        return "tflite tests skipped"
    if _SKIP_TF and test.model_type not in ["tflite", "tfjs"]:
        return "tf tests skipped"
    ok, reason = test.check_opset_constraints(_OPSET)
    if not ok:
        return reason
    if test.tf_min_version and tf_utils.get_tf_version() < Version(str(test.tf_min_version)):
        return f"requires TF >= {test.tf_min_version}"
    return None


@pytest.mark.parametrize("name", _TEST_IDS)
def test_pretrained_model(name):
    test = _TESTS[name]
    reason = _skip_reason(test)
    if reason:
        pytest.skip(reason)
    result = test.run_test(name, backend=_BACKEND, opset=_OPSET)
    assert result, f"Model {name!r} failed conversion or validation"
