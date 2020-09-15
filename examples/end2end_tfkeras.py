import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input


# Creates the model.
model = keras.Sequential()
#model.add(layers.Embedding(input_dim=10, output_dim=4))
model.add(Input((4, 4)))
model.add(layers.SimpleRNN(8))
model.add(layers.Dense(2))
print(model.summary())
print(model.inputs)
print(model.outputs)

# Testing the model.
input = np.random.randn(2, 4, 4).astype(np.float32)
expected  = model.predict(input)
print(expected)

# Training
# ....

# Saves the model.
if not os.path.exists("simple_rnn"):
    os.mkdir("simple_rnn")
tf.keras.models.save_model(model, "simple_rnn")

# Run the command line.
proc = subprocess.run('python -m tf2onnx.convert --saved-model simple_rnn '
                      '--output simple_rnn.onnx --opset 12'.split(),
                      capture_output=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

# Run onnxruntime.
from onnxruntime import InferenceSession
session = InferenceSession("simple_rnn.onnx")
got = session.run(None, {'input_1:0': input})
print(got[0])

# Differences
print(np.abs(got[0] - expected).max())
