# SPDX-License-Identifier: Apache-2.0


"""Unit tests using onnx backends."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=missing-docstring,unused-import

import os
import zipfile

import numpy as np
import tensorflow as tf

from common import check_tf_min_version, unittest_main
from backend_test_base import Tf2OnnxBackendTestBase
import tf2onnx


class ApiTests(Tf2OnnxBackendTestBase):
    """Test tf2onnx python API."""

    def setUp(self):
        super().setUp()
        os.makedirs(self.test_data_directory, exist_ok=True)

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

    def _test_keras_api(self, large_model=False):
        model = self.create_model()
        shape = [1, 224, 224, 3]
        x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        ky = model(x)
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        if large_model:
            output_path = os.path.join(self.test_data_directory, "model.zip")
        else:
            output_path = os.path.join(self.test_data_directory, "model.onnx")

        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=self.config.opset, large_model=large_model, output_path=output_path)
        # input_names = [n.name for n in model_proto.graph.input]
        output_names = [n.name for n in model_proto.graph.output]
        if large_model:
            # need to unpack the zip for run_onnxruntime()
            with zipfile.ZipFile(output_path, 'r') as z:
                z.extractall(os.path.dirname(output_path))
            output_path = os.path.join(os.path.dirname(output_path), "__MODEL_PROTO.onnx")

        oy = self.run_onnxruntime(output_path, {"input:0": x}, output_names)
        self.assertAllClose(ky.numpy(), oy[0], rtol=0.3, atol=0.1)
        # make sure the original keras model wasn't trashed
        ky1 = model(x)
        self.assertAllClose(ky1.numpy(), oy[0], rtol=0.3, atol=0.1)

    @check_tf_min_version("2.0")
    def test_keras_api(self):
        self._test_keras_api(large_model=False)

    @check_tf_min_version("2.0")
    def test_keras_api_large(self):
        self._test_keras_api(large_model=True)

    @check_tf_min_version("2.0")
    def test_function(self):
        def func(x, y):
            return x * y

        output_path = os.path.join(self.test_data_directory, "model.onnx")
        shape = [1, 10, 10]
        x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        y = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

        spec = (tf.TensorSpec((None, 10, 10), tf.float32, name="x"),
                tf.TensorSpec((None, 10, 10), tf.float32, name="y"))
        concrete_func = tf.function(func, input_signature=spec)

        ky = func(x, y)

        model_proto, _ = tf2onnx.convert.from_function(concrete_func, input_signature=spec,
                                                       opset=self.config.opset, output_path=output_path)
        output_names = [n.name for n in model_proto.graph.output]
        oy = self.run_onnxruntime(output_path, {"x:0": x, "y:0": y}, output_names)
        self.assertAllClose(ky, oy[0], rtol=0.3, atol=0.1)

    @check_tf_min_version("1.15")
    def test_graphdef(self):
        def func(x, y):
            return x * y

        output_path = os.path.join(self.test_data_directory, "model.onnx")
        shape = [1, 10, 10]
        x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        y = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        ky = func(x, y)

        # make a graphdef
        spec = (tf.TensorSpec((None, 10, 10), tf.float32, name="x"),
                tf.TensorSpec((None, 10, 10), tf.float32, name="y"))
        function = tf.function(func, input_signature=spec)
        concrete_func = function.get_concrete_function(*spec)
        graph_def = concrete_func.graph.as_graph_def(add_shapes=True)

        model_proto, _ = tf2onnx.convert.from_graph(graph_def, input_names=["x:0", "y:0"], output_names=["Identity:0"],
                                                    opset=self.config.opset, output_path=output_path)
        output_names = [n.name for n in model_proto.graph.output]
        oy = self.run_onnxruntime(output_path, {"x:0": x, "y:0": y}, output_names)
        self.assertAllClose(ky, oy[0], rtol=0.3, atol=0.1)


if __name__ == '__main__':
    unittest_main()
