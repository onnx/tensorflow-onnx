# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
import numpy
import numpy.random as rnd
from _tools import generate_text_inputs, benchmark


def main(opset=13):
    
    if False:
        import tensorflow as tf
        import tensorflow_text
        import tensorflow_hub as hub
        sentences = tf.constant(["Hi I'm some text"])
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/lambert_en_uncased_L-24_H-1024_A-16/2", trainable=True)
        preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder_inputs = preprocessor(text_input)
        embedded_inputs = {k: v.numpy() for k, v in preprocessor(sentences).items()}
        for k, v in embedded_inputs.items():
            print(k, v.dtype, v.shape)
    
    url = "https://tfhub.dev/tensorflow/lambert_en_uncased_L-24_H-1024_A-16/2?tf-hub-format=compressed"
    dest = "tf-lambert_en_uncased_L-24_H-1024_A-16"
    name = "lambert_en_uncased_L-24_H-1024_A-16"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    inputs = [OrderedDict([
        ('input_word_ids', numpy.array([rnd.randint(0, 1000) for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1))),
        ('input_mask', numpy.array([rnd.randint(0, 1) for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1))),
        ('input_type_ids', numpy.array([i//5 for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1)))
    ]) for i in range(0, 10)]

    benchmark(url, dest, onnx_name, opset, inputs, output_name="pooled_output")


if __name__ == "__main__":
    main()
