# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for string ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
from tensorflow_text.python.ops import regex_split_ops
from backend_test_base import Tf2OnnxBackendTestBase
from common import requires_custom_ops
from tf2onnx import utils, constants

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
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        sh = np.array([2, 3], dtype=np.int64)
        dv = np.array(-1, dtype=np.int64)

        def func(x, dv, sh):
            r = tf.RaggedTensor.from_tensor(x, padding=-1)
            x_ = tf.raw_ops.RaggedTensorToTensor(
                shape=sh, values=r.flat_values, default_value=dv,
                row_partition_tensors=[r.row_splits],
                row_partition_types=["ROW_SPLITS"])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: x, _INPUT1: dv, _INPUT2: sh},
                            as_session=False)

    @requires_custom_ops("RaggedTensorToTensorOp")
    def test_regex_split_ragged_tensor(self):
        x = np.array(["ab abc bc"], dtype=np.str)

        def func(x):
            actual_tokens, start, end = regex_split_ops.regex_split_with_offsets(
                input=x, delim_regex_pattern="(\\s)", keep_delim_regex_pattern="")
            shape = actual_tokens.shape
            return tf.identity(shape, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x})


if __name__ == "__main__":
    unittest.main()
