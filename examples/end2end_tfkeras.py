# SPDX-License-Identifier: Apache-2.0

"""
This example builds a simple model without training.
It is converted into ONNX directly, by serializing to byte string.
Predictions are compared to the predictions from tensorflow to check there is no
discrepencies. Inferencing time is also compared between
*onnxruntime*, *tensorflow* and *tensorflow.lite*.
"""


from onnxruntime import InferenceSession
import timeit
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Input

########################################
# Creates the model.
model = keras.Sequential()
model.add(Input((4, 4)))
model.add(layers.SimpleRNN(8))
model.add(layers.Dense(2))
print(model.summary())
input_names = [n.name for n in model.inputs]
output_names = [n.name for n in model.outputs]
print('inputs:', input_names)
print('outputs:', output_names)

########################################
# Training
# ....
# Skipped.

########################################
# Testing the model.
input = np.random.randn(2, 4, 4).astype(np.float32)
expected = model.predict(input)
print(expected)

########################################
# Serialize but do not save the model
from tf2onnx.keras2onnx_api import convert_keras
onnx_model = convert_keras(model=model,name='example')
onnx_model_as_byte_string = onnx_model.SerializeToString()

########################################
# Runs onnxruntime.
session = InferenceSession(onnx_model_as_byte_string)
got = session.run(None, {'input_1': input})
print(got[0])

########################################
# Measures the differences.
print(np.abs(got[0] - expected).max())

########################################
# Measures processing time.
print('tf:', timeit.timeit('model.predict(input)',
                           number=100, globals=globals()))
print('ort:', timeit.timeit("session.run(None, {'input_1': input})",
                            number=100, globals=globals()))
