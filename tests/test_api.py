# SPDX-License-Identifier: Apache-2.0


"""Unit tests using onnx backends."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
# pylint reports unused-wildcard-import which is false positive, __all__ is defined in common
import tf2onnx
from tf2onnx import constants, utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.tf_loader import is_tf2, tf_placeholder_with_default, tf_placeholder


class ApiTests(Tf2OnnxBackendTestBase):
    def setUp(self):
        super(ApiTests, self).setUp()
        self.temp_dir = utils.get_temp_directory()
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        utils.delete_directory(self.temp_dir)

    def create_model(self):
        strides = 2
        inp = tf.keras.Input(shape=[None, None, 3], name="input")
        conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=strides, name="conv1")(inp)
        conv = tf.keras.layers.BatchNormalization()(conv)
        model = tf.keras.Model(inputs=inp, outputs=conv,
                               name="test_model", trainable=True)
        optimizer = tf.keras.optimizers.Adam(1e-4, 1e-8)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        return model

    @check_tf_min_version("1.15")
    def test_keras_api(self):
        model = self.create_model()
        shape = [1, 224, 224, 3]
        x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        ky = model(x)
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=self.config.opset, large_model=True)
        model_path = tf2onnx.utils.save_onnx_model(
            self.temp_dir, "model.onnx", None, model_proto, external_tensor_storage=external_tensor_storage)
        # input_names = [n.name for n in model_proto.graph.input]
        output_names = [n.name for n in model_proto.graph.output]
        oy = self.run_onnxruntime(model_path, {"input:0": x}, output_names)
        self.assertAllClose(ky.numpy(), oy[0], rtol=0.3, atol=0.1)
        # make sure the original keras model wasn't trashed
        ky1 = model(x)
        self.assertAllClose(ky1.numpy(), oy[0], rtol=0.3, atol=0.1)

    @check_tf_min_version("1.15")
    def test_function(self):
        pass

    @check_tf_min_version("1.15")
    def test_graphdef(self):
        pass


if __name__ == '__main__':
    unittest_main()
