# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for string ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import requires_custom_ops
from tf2onnx import utils
from tf2onnx import constants

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,import-outside-toplevel
# pylint: disable=wrong-import-position

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"
_TFOUTPUT2 = "output2"
_OUTPUT2 = "output2:0"


class VariantOpsTests(Tf2OnnxBackendTestBase):

    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
        process_args = {"extra_opset": extra_opset}
        return self.run_test_case(func, feed_dict, [], output_names_with_port, process_args=process_args, **kwargs)

    def run_onnxruntime(self, model_path, inputs, output_names):
        """Run test against onnxruntime backend."""
        from onnxruntime_customops import get_library_path
        import onnxruntime as rt
        opt = rt.SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        m = rt.InferenceSession(model_path, opt)
        results = m.run(output_names, inputs)
        return results

    @requires_custom_ops("RaggedTensorToTensorOp")
    def test_ragged_tensor(self):
        digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
        
        def func():
            x_ = digits.to_tensor()
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {})


if __name__ == "__main__":
    unittest.main()
