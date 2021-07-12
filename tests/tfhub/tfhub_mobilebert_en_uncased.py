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
        preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder_inputs = preprocessor(text_input)
        embedded_inputs = {k: v.numpy() for k, v in preprocessor(sentences).items()}
        for k, v in embedded_inputs.items():
            print(k, v.dtype, v.shape)
            
    url = "https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1?tf-hub-format=compressed"
    dest = "tf-mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT"
    name = "mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    inputs = generate_text_inputs()
    benchmark(url, dest, onnx_name, opset, inputs,
              output_name="attention_scores")  #, ort_name="mobile_bert_encoder_50")


if __name__ == "__main__":
    main()
