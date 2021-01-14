# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for string ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

class StringOpsTests(Tf2OnnxBackendTestBase):

    @requires_custom_ops("StringRegexReplace")
    def test_static_regex_replace(self):
        text_val = np.array([["Hello world!", "Test 1 2 3"], ["Hi there", "test test"]], dtype=np.str)
        def func(text):
            x_ = tf.strings.regex_replace(text, " ", "_", replace_global=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringJoin")
    def test_string_join(self):
        text_val1 = np.array([["a", "Test 1 2 3"], ["Hi there", "test test"]], dtype=np.str)
        text_val2 = np.array([["b", "Test 1 2 3"], ["Hi there", "suits ♠♣♥♦"]], dtype=np.str)
        text_val3 = np.array("Some scalar text", dtype=np.str)
        def func(text1, text2, text3):
            x_ = tf.strings.join([text1, text2, text3], separator="±")
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1, _INPUT1: text_val2, _INPUT2: text_val3})

    @requires_custom_ops("StringSplit")
    def test_string_split(self):
        text_val = np.array([["a", "Test 1 2 3"], ["Hi there", "test test"]], dtype=np.str)
        def func(text):
            x = tf.strings.split(text, sep=' ').flat_values
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringToHashBucketFast")
    def test_string_to_hash_bucket_fast(self):
        text_val = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=np.str)
        def func(text):
            x = tf.strings.to_hash_bucket_fast(text, 20)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringEqual")
    def test_string_equal(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=np.str)
        text_val2 = np.array([["a", "Test 2 4 6", "♠♣"], ["Hello", "test test", "♥ ♦"]], dtype=np.str)
        def func(text1, text2):
            x = tf.equal(text1, text2)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1, _INPUT1: text_val2})

    @requires_custom_ops("StringNotEqual")
    def test_string_not_equal(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=np.str)
        text_val2 = np.array([["a", "Test 2 4 6", "♠♣"], ["Hello", "test test", "♥ ♦"]], dtype=np.str)
        def func(text1, text2):
            x = tf.not_equal(text1, text2)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1, _INPUT1: text_val2})

    # Make sure that fallback works for non-string equality
    @requires_custom_ops()
    def test_equal(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @requires_custom_ops()
    def test_not_equal(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.not_equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
        process_args = {"extra_opset": extra_opset}
        return self.run_test_case(func, feed_dict, [], output_names_with_port, process_args=process_args, **kwargs)

    def run_onnxruntime(self, model_path, inputs, output_names):
        """Run test against onnxruntime backend."""
        from ortcustomops import get_library_path
        import onnxruntime as rt
        opt = rt.SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        m = rt.InferenceSession(model_path, opt)
        results = m.run(output_names, inputs)
        return results
