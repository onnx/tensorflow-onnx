# SPDX-License-Identifier: Apache-2.0


"""Unit tests using onnx backends."""

import os
import unittest
from distutils.version import LooseVersion
from itertools import product

import numpy as np
from numpy.testing import assert_almost_equal
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from backend_test_base import Tf2OnnxBackendTestBase
# pylint reports unused-wildcard-import which is false positive, __all__ is defined in common
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tf2onnx import constants, utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.tf_loader import is_tf2, tf_placeholder_with_default, tf_placeholder
from tf2onnx.onnx_opset.signal import make_dft_constant

# pylint: disable=missing-docstring,invalid-name,unused-argument,function-redefined,cell-var-from-loop


# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_INPUT1 = "input1:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"


class BugTests(Tf2OnnxBackendTestBase):

    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        kwargs["convert_var_to_const"] = False
        kwargs["constant_fold"] = False
        return self.run_test_case(func, feed_dict, [], output_names_with_port, **kwargs)

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_abc_cde_abde(self):
        x_val = np.random.random([2, 3, 5]).astype(np.float32)
        y_val = np.random.random([5, 7, 11]).astype(np.float32)
        def func(x, y):
            ret = tf.einsum("abc,cde->abde", x, y)
            return tf.identity(ret, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_abcd_cde_abe(self):
        x_val = np.random.random([2, 3, 5, 7]).astype(np.float32)
        y_val = np.random.random([5, 7, 11]).astype(np.float32)
        def func(x, y):
            ret = tf.einsum("abcd,cde->abe", x, y)
            return tf.identity(ret, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    def common_save_model_einsum(self, equation, x_val, y_val):
        @tf.function
        def conv_model(x, y):
            return tf.einsum(equation, x, y)

        class MyModule(tf.Module):
            @tf.function
            def __call__(self, x, y):
                return conv_model(x, y)

        module_func = MyModule()
        module_func(x_val, y_val)    
        tf.saved_model.save(module_func, "einsum_" + equation.replace(",", "_").replace("->", "__"))

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_abc_cde_abde_2(self):
        x_val = np.random.random([2, 3, 5]).astype(np.float32)
        y_val = np.random.random([5, 7, 11]).astype(np.float32)
        self.common_save_model_einsum("abc,cde->abde", x_val, y_val)
        # convert with
        # python -m tf2onnx.convert --saved-model einsum_m1 --output einsum1.onnx --opset 13 --tag serve --concrete_function 0

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_abcd_cde_abe_2(self):
        x_val = np.random.random([2, 3, 5, 7]).astype(np.float32)
        y_val = np.random.random([5, 7, 11]).astype(np.float32)
        self.common_save_model_einsum("abcd,cde->abe", x_val, y_val)
        # convert with
        # python -m tf2onnx.convert --saved-model einsum_m1 --output einsum1.onnx --opset 13 --tag serve --concrete_function 0


if __name__ == '__main__':
    unittest_main()
