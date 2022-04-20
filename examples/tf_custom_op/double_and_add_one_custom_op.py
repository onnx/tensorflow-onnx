# SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
import onnx
import os
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.tf_loader import tf_placeholder


DIR_PATH = os.path.realpath(os.path.dirname(__file__))
saved_model_path = os.path.join(DIR_PATH, "model.onnx")
tf_library_path = os.path.join(DIR_PATH, "double_and_add_one.so")


@tf_op("DoubleAndAddOne")
class DoubleAndAddOne:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Mul"
        node_shape = ctx.get_shape(node.input[0])
        node_dtype = ctx.get_dtype(node.input[0])
        node_np_dtype = utils.map_onnx_to_numpy_type(node_dtype)

        const_two = ctx.make_const(utils.make_name("const_two"), np.array([2]).astype(node_np_dtype)).output[0]
        node.input.append(const_two)

        const_one = ctx.make_const(utils.make_name("const_one"), np.ones(node_shape, dtype=node_np_dtype)).output[0]
        op_name = utils.make_name(node.name)
        ctx.insert_new_node_on_output("Add", node.output[0], inputs=[node.output[0], const_one], name=op_name)


@tf.function
def func(x):
    custom_op = tf.load_op_library(tf_library_path)
    x_ = custom_op.double_and_add_one(x)
    output = tf.identity(x_, name="output")
    return output

spec = [tf.TensorSpec(shape=(2, 3), dtype=tf.int32, name="input")]

onnx_model, _ = tf2onnx.convert.from_function(func, input_signature=spec, opset=15)

with open(saved_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

onnx_model = onnx.load(saved_model_path)
onnx.checker.check_model(onnx_model)


## Run the model in ONNXRuntime to verify the result.
import onnxruntime as ort
input = np.arange(6).reshape(2,3).astype(np.int32)
ort_session = ort.InferenceSession(saved_model_path)
ort_inputs = {ort_session.get_inputs()[0].name: input}

ort_outs = ort_session.run(None, ort_inputs)
print("input:", input, "\nort_outs:", ort_outs)

'''
input: [[0 1 2]
        [3 4 5]] 
ort_outs: [array([[ 1,  3,  5],
                  [ 7,  9, 11]], dtype=int32)]
'''
