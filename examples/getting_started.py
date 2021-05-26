# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to convert tf functions and keras models using the Python API.
It also demonstrates converting saved_models from the command line.
"""

import tensorflow as tf
import tf2onnx
import numpy as np
import onnxruntime as ort
import os

##################### tf function #####################

@tf.function
def f(a, b):
    return a + b

input_signature = [tf.TensorSpec([2, 3], tf.float32), tf.TensorSpec([2, 3], tf.float32)]
onnx_model, _ = tf2onnx.convert.from_function(f, input_signature, opset=13)

a_val = np.ones([2, 3], np.float32)
b_val = np.zeros([2, 3], np.float32)

print("Tensorflow result")
print(f(a_val, b_val).numpy())

print("ORT result")
sess = ort.InferenceSession(onnx_model.SerializeToString())
res = sess.run(None, {'a': a_val, 'b': b_val})
print(res[0])


##################### Keras Model #####################

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation="relu"))

input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

x_val = np.ones((3, 3), np.float32)

print("Keras result")
print(model(x_val).numpy())

print("ORT result")
sess = ort.InferenceSession(onnx_model.SerializeToString())
res = sess.run(None, {'x': x_val})
print(res[0])


##################### Saved Model #####################

model.save("savedmodel")
os.system("python -m tf2onnx.convert --saved-model savedmodel --output model.onnx --opset 13")

print("ORT result")
sess = ort.InferenceSession("model.onnx")
res = sess.run(None, {'dense_input': x_val})
print(res[0])

print("Conversion succeeded")