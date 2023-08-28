# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for string ops."""

import unittest
import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import requires_custom_ops, check_tf_min_version, check_opset_min_version
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
_TFOUTPUT3 = "output3"
_OUTPUT3 = "output3:0"

class StringOpsTests(Tf2OnnxBackendTestBase):

    @requires_custom_ops("StringRegexReplace")
    def test_static_regex_replace(self):
        text_val = np.array([["Hello world!", "Test 1 2 3"], ["Hi there", "test test"]], dtype=str)
        def func(text):
            x_ = tf.strings.regex_replace(text, " ", "_", replace_global=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringJoin")
    @check_opset_min_version(8, "Expand")
    def test_string_join(self):
        text_val1 = np.array([["a", "Test 1 2 3"], ["Hi there", "test test"]], dtype=str)
        text_val2 = np.array([["b", "Test 1 2 3"], ["Hi there", "suits ♠♣♥♦"]], dtype=str)
        text_val3 = np.array("Some scalar text", dtype=str)
        def func(text1, text2, text3):
            x_ = tf.strings.join([text1, text2, text3], separator="±")
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1, _INPUT1: text_val2, _INPUT2: text_val3})

    @requires_custom_ops("ReduceJoin")
    def test_reduce_join(self):
        text_val = np.array([["a", "Test 1 2 3"], ["b", "test test"], ["c", "Hi there Test"]], dtype=str)
        def func(text):
            x_ = tf.strings.reduce_join(text, axis=1, separator="±")
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringSplit")
    @check_tf_min_version("2.0", "result is sparse not ragged in tf1")
    def test_string_split(self):
        text_val = np.array([["a", "Test 1 2 3"], ["Hi there", "test test"]], dtype=str)
        def func(text):
            x = tf.strings.split(text, sep=' ').flat_values
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringToHashBucketFast")
    def test_string_to_hash_bucket_fast(self):
        text_val = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=str)
        def func(text):
            x = tf.strings.to_hash_bucket_fast(text, 20)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val})

    @requires_custom_ops("StringEqual")
    def test_string_equal(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=str)
        text_val2 = np.array([["a", "Test 2 4 6", "♠♣"], ["Hello", "test test", "♥ ♦"]], dtype=str)
        def func(text1, text2):
            x = tf.equal(text1, text2)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1, _INPUT1: text_val2})

    @requires_custom_ops("StringNotEqual")
    def test_string_not_equal(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=str)
        text_val2 = np.array([["a", "Test 2 4 6", "♠♣"], ["Hello", "test test", "♥ ♦"]], dtype=str)
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

    @requires_custom_ops("RegexSplitWithOffsets")
    @check_tf_min_version("2.3", "tensorflow_text")
    def test_regex_split_with_offsets(self):
        from tensorflow_text.python.ops.regex_split_ops import (
            gen_regex_split_ops as lib_gen_regex_split_ops)
        text_val = np.array(["a Test 1 2 3 ♠♣",
                             "Hi there test test ♥♦"], dtype=str)
        def func(text):
            tokens, begin_offsets, end_offsets, row_splits = lib_gen_regex_split_ops.regex_split_with_offsets(
                text, "(\\s)", "")
            tokens_ = tf.identity(tokens, name=_TFOUTPUT)
            begin_ = tf.identity(begin_offsets, name=_TFOUTPUT1)
            end_ = tf.identity(end_offsets, name=_TFOUTPUT2)
            rows_ = tf.identity(row_splits, name=_TFOUTPUT3)
            return tokens_, begin_, end_, rows_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3], {_INPUT: text_val})

    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
        process_args = {"extra_opset": extra_opset}
        return self.run_test_case(func, feed_dict, [], output_names_with_port,
                                  use_custom_ops=True, process_args=process_args, **kwargs)

    @requires_custom_ops("WordpieceTokenizer")
    @check_tf_min_version("2.3", "tensorflow_text")
    @unittest.skip("Not fixed yet")
    def test_wordpiece_tokenizer(self):
        from tensorflow_text.python.ops.wordpiece_tokenizer import (
            gen_wordpiece_tokenizer as lib_gen_wordpiece_tokenizer)
        from tensorflow.python.ops.ragged import ragged_tensor
        from tensorflow.python.ops import lookup_ops

        def _CreateTable(vocab, num_oov=1):
            init = tf.lookup.KeyValueTensorInitializer(
                vocab,
                tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
                key_dtype=tf.string,
                value_dtype=tf.int64,
                name="hasht")
            return lookup_ops.StaticVocabularyTable(
                init, num_oov, lookup_key_dtype=tf.string)

        vocab = _CreateTable(["great", "they", "the", "##'", "##re", "##est"])
        text_val = np.array(["they're", "the", "greatest"], dtype=str)

        def func(text):
            inputs = ragged_tensor.convert_to_tensor_or_ragged_tensor(text)
            result = lib_gen_wordpiece_tokenizer.wordpiece_tokenize_with_offsets(
                inputs, vocab.resource_handle, "##", 200, True, "[UNK]")
            tokens, begin_offsets, end_offsets, rows = result
            tokens_ = tf.identity(tokens, name=_TFOUTPUT)
            begin_ = tf.identity(begin_offsets, name=_TFOUTPUT1)
            end_ = tf.identity(end_offsets, name=_TFOUTPUT2)
            rows_ = tf.identity(rows, name=_TFOUTPUT3)
            return tokens_, begin_, end_, rows_
        # Fails due to Attempting to capture an EagerTensor without building a function.
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3],
                            {_INPUT: text_val}, as_session=True)


if __name__ == "__main__":
    unittest.main()
