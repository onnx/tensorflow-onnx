# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for custom rnns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops import init_ops, random_ops, init_ops
from tensorflow.python.ops.array_ops import FakeQuantWithMinMaxVars
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow_model_optimization.quantization.keras import quantize_model
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


from keras import backend as K
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from keras2onnx import convert_keras

QuantizeAwareActivation = quantize_aware_activation.QuantizeAwareActivation
QuantizeWrapper = quantize_wrapper.QuantizeWrapper
QuantizeRegistry = default_8bit_quantize_registry.QuantizeRegistry

keras = tf.keras
layers = tf.keras.layers

custom_object_scope = tf.keras.utils.custom_object_scope
deserialize_layer = tf.keras.layers.deserialize
serialize_layer = tf.keras.layers.serialize



def freeze_session(graph, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param graph The TensorFlow graph to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    
    Source: https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
    """
    with graph.as_default():
        freeze_var_names = list(set(
            v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


class QuantizationTests(Tf2OnnxBackendTestBase):
 
    def setUp(self):
        super(QuantizationTests, self).setUp()
        self.quantize_registry = QuantizeRegistry()
    
    def test_quantize_keras(self):
        model = quantize_model(
            keras.Sequential([
                layers.Dense(3, activation='relu', input_shape=(5,)),
                layers.Dense(3, activation='relu', input_shape=(3,)),
            ]))
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        print(model.summary())
        x = np.array([[0, 1, 2, 3, 4]], dtype=np.float32)
        y = model.predict(x)
        print(y)
        model_onnx = convert_keras(model)
        print(model_onnx)

    def test_quantize_tf(self):
        inputs = tf.keras.layers.Input(shape=(5,))
        inter = tf.keras.layers.Dense(3, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(3, activation='relu')(inter)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        model = quantize_model(model)
        print(model.summary())
        x = np.array([[0, 1, 2, 3, 4]], dtype=np.float32)
        y = model(x)
        print(y)
        print(type(model))
        
        frozen_graph = freeze_session(
                model,
                output_names=[out.op.name for out in model.outputs])

        sess.graph.as_default()
        softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')
        predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': x})

        model_onnx = convert_keras(model)
        assert model_onnx is not None

    def testQuantizesOutputsFromLayer(self):
        # source https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_wrapper_test.py        
        # TODO(pulkitb): Increase coverage by adding other output quantize layers
        # such as AveragePooling etc.

        layer = layers.ReLU()
        quantized_model = keras.Sequential([
            QuantizeWrapper(
                layers.ReLU(),
                quantize_config=self.quantize_registry.get_quantize_config(layer))
        ])

        model = keras.Sequential([layers.ReLU()])

        inputs = np.random.rand(1, 2, 1)
        expected_output = tf.quantization.fake_quant_with_min_max_vars(
            model.predict(inputs), -6.0, 6.0, num_bits=8, narrow_range=False)
        exp = quantized_model.predict(inputs)
        self.assertAllClose(expected_output, exp)

        model_onnx = convert_keras(model)
        quantized_model_onnx = convert_keras(quantized_model)
        assert model_onnx is not None
        assert quantized_model_onnx is not None
        if 'op_type: "FakeQuantWithMinMaxVars"' in str(quantized_model_onnx):
            raise AssertionError(
                "FakeQuantWithMinMaxVars not replaced\n{}".format(quantized_model_onnx))
        assert 'op_type: "QuantizeLinear"' in str(quantized_model_onnx)
        assert 'op_type: "DequantizeLinear"' in str(quantized_model_onnx)
        from onnxruntime import InferenceSession
        sess = InferenceSession(quantized_model_onnx.SerializeToString())
        names = [_.name for _ in sess.get_inputs()]        
        got = sess.run(None, {names[0]: x})
        self.assertAllClose(expected_output, got)
    
    
if __name__ == '__main__':
    unittest_main()
