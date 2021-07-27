# SPDX-License-Identifier: Apache-2.0
import os
import numpy
import onnxruntime as ort
import tensorflow as tf
import tensorflow_hub as hub
import tf2onnx
from _tools import generate_random_images, check_discrepencies

imgs = generate_random_images(shape=(1, 224, 224, 3), scale=1.)

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/5",
                   trainable=False)])
model.build([None, 224, 224, 3])

expected_output = model(imgs[0])

dest = "tf-resnet_v1_101"
if not os.path.exists(dest):
    os.makedirs(dest)
dest_name = os.path.join(dest, "resnet_v1_101-13-keras.onnx")
if not os.path.exists(dest_name):
    tf2onnx.convert.from_keras(model, opset=13, output_path=dest_name)

sess = ort.InferenceSession(dest_name)
print('inputs', [_.name for _ in sess.get_inputs()])
ort_output = sess.run(None, {"keras_layer_input": imgs[0]})

print("Actual")
print(ort_output)
print("Expected")
print(expected_output)

diff = expected_output.numpy() - ort_output[0]
max_diff = numpy.abs(diff).max()
rel_diff = (numpy.abs(diff) / (expected_output.numpy() + 1e-5)).max()
print(max_diff, rel_diff, [ort_output[0].min(), ort_output[0].max()])
