# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for custom rnns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops, random_ops, init_ops
from tensorflow.python.ops.array_ops import FakeQuantWithMinMaxVars
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_gru_count, check_opset_after_tf_version, skip_tf2
from tf2onnx.tf_loader import is_tf2
from tensorflow_model_optimization.python.core.quantization.keras import quantize

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ

if is_tf2():
    fake_quant_with_min_max_vars_gradient = tf.compat.v1.quantization.fake_quant_with_min_max_vars_gradient
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
else:
    fake_quant_with_min_max_vars_gradient = tf.quantization.fake_quant_with_min_max_vars_gradient
    dynamic_rnn = tf.nn.dynamic_rnn


def quantize_model_save(keras_file, tflite_file):
    with quantize.quantize_scope():
        model = tf.keras.models.load_model(keras_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.representative_dataset = calibration_gen
    converter._experimental_new_quantizer = True  # pylint: disable=protected-access
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]  # to enable post-training quantization with the representative dataset

    tflite_model = converter.convert()
    tflite_file = 'quantized_mnist.tflite'
    open(tflite_file, 'wb').write(tflite_model)


class QuantizationTests(Tf2OnnxBackendTestBase):
    
    def common_quantize(self, name):
        dest = os.path.splitext(os.path.split(name)[-1])[0] + '.tflite'
        quantize_model_save(name, dest)
        

    def test_fake_quant_with_min_max_vars_gradient(self):
        cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        name = os.path.join(cwd, "gru", "frozen.pb")
        self.common_quantize(name)


if __name__ == '__main__':
    unittest_main()
