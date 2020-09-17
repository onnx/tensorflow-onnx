"""
This example builds a simple model without training.
It is converted into ONNX. Predictions are compared to
the predictions from tensorflow to check there is no
discrepencies. Inferencing time is also compared between
*onnxruntime*, *tensorflow* and *tensorflow.lite*.
"""
from onnxruntime import InferenceSession
import os
import subprocess
import timeit
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tf2onnx.tf_loader import tf_reset_default_graph, tf_session, freeze_session

########################################
# Creates the model.
model = keras.Sequential()
#model.add(layers.Embedding(input_dim=10, output_dim=4))
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
# Saves the model.
if not os.path.exists("simple_rnn"):
    os.mkdir("simple_rnn")
tf.keras.models.save_model(model, "simple_rnn")

########################################
# Run the command line.
proc = subprocess.run('python -m tf2onnx.convert --saved-model simple_rnn '
                      '--output simple_rnn.onnx --opset 12'.split(),
                      capture_output=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

########################################
# Runs onnxruntime.
session = InferenceSession("simple_rnn.onnx")
got = session.run(None, {'input_1:0': input})
print(got[0])

########################################
# Measures the differences.
print(np.abs(got[0] - expected).max())

########################################
# Measures processing time.
print('tf:', timeit.timeit('model.predict(input)',
                           number=100, globals=globals()))
print('ort:', timeit.timeit("session.run(None, {'input_1:0': input})",
                            number=100, globals=globals()))

########################################
# Freezes the graph with tensorflow.lite
converter = tf.lite.TFLiteConverter.from_saved_model("simple_rnn")
tflite_model = converter.convert()
with open("simple_rnn.tflite", "wb") as f:
    f.write(tflite_model)

# Builds an interpreter
interpreter = tf.lite.Interpreter(model_path='simple_rnn.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details", input_details)
print("output_details", output_details)
index = input_details[0]['index']


def tflite_predict(input, interpreter=interpreter, index=index):
    res = []
    for i in range(input.shape[0]):
        interpreter.set_tensor(index, input[i:i + 1])
        interpreter.invoke()
        res.append(interpreter.get_tensor(output_details[0]['index']))
    return np.vstack(res)


print(input[0:1].shape, "----", input_details[0]['shape'])
output_data = tflite_predict(input, interpreter, index)
print(output_data)

########################################
# Measures processing time again.

print('tf:', timeit.timeit('model.predict(input)',
                           number=100, globals=globals()))
print('ort:', timeit.timeit("session.run(None, {'input_1:0': input})",
                            number=100, globals=globals()))
print('tflite:', timeit.timeit('tflite_predict(input)',
                               number=100, globals=globals()))

########################################
# Measures processing time only between onnxruntime and
# tensorflow lite with more loops.

print('ort:', timeit.timeit("session.run(None, {'input_1:0': input})",
                            number=10000, globals=globals()))
print('tflite:', timeit.timeit('tflite_predict(input)',
                               number=10000, globals=globals()))
