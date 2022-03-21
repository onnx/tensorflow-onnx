# SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import os
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.tf_loader import tf_placeholder


DIR_PATH = os.path.realpath(os.path.dirname(__file__))
saved_model_path = os.path.join(DIR_PATH, "model.onnx")
tf_library_path = os.path.join(DIR_PATH, "add_one.so")


@tf_op("AddOne", onnx_op="Add")
class AddOne:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node_shape = ctx.get_shape(node.input[0])
        const_one = ctx.make_const(utils.make_name("const_one"), np.ones(node_shape, dtype = np.int32)).output[0]
        node.input.append(const_one)

@tf.function
def func(x):
    AddOne = tf.load_op_library(tf_library_path)
    x_ = AddOne.add_one(x)
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
