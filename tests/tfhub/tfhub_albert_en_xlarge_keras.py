# Adapted the sample code on https://tfhub.dev/tensorflow/albert_en_xlarge/3
import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub

# Using hub.load instead of KerasLayer lets us easily intercept the results of the
# preprocessor before passing it to the encoder
preprocessor = hub.load("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
encoder = hub.load("https://tfhub.dev/tensorflow/albert_en_xlarge/3")
sentences = tf.constant(["Hi I'm some text"])

embedded_inputs = {k: v.numpy() for k, v in preprocessor(sentences).items()}
print("Inputs")
print(embedded_inputs)
expected_output = encoder(embedded_inputs)["pooled_output"].numpy()

# Now make an actual keras layer for the part we want to convert
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/albert_en_xlarge/3",
    trainable=True)

# To convert it to a model, we need the input shapes/types. These can be
# determined from the types/shapes/names of embedded_inputs. Remove the batch dim from the shapes.
encoder_inputs = {
    "input_word_ids": tf.keras.Input(shape=[None], dtype="int32", name="input_word_ids"),
    "input_mask": tf.keras.Input(shape=[None], dtype="int32", name="input_mask"),
    "input_type_ids": tf.keras.Input(shape=[None], dtype="int32", name="input_type_ids"),
}
encoder_outputs = encoder(encoder_inputs)["pooled_output"]
encoding_model = tf.keras.Model(encoder_inputs, encoder_outputs)


import tf2onnx
import onnxruntime as ort
import zipfile
import os
print("Converting")

dest = "tf-albert-en-xlarge"
if not os.path.exists(dest):
    os.makedirs(dest)
dest_name = os.path.join(dest, "albert_en_xlarge.zip")

# This model is a large model.
tf2onnx.convert.from_keras(encoding_model, opset=13, large_model=True, output_path=dest_name)
# To run the model in ORT we need to unzip it.
with zipfile.ZipFile(dest_name, 'r') as z:
  z.extractall(os.path.join(dest, "albert_en_xlarge"))
sess = ort.InferenceSession(os.path.join(dest, "albert_en_xlarge", "__MODEL_PROTO.onnx"))
ort_output = sess.run(None, embedded_inputs)
print("Actual")
print(ort_output[0])
print("Expected")
print(expected_output)
