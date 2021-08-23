# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark
import tf2onnx
import onnxruntime as ort


def main(opset=13):
    url = "https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1?tf-hub-format=compressed"
    dest = "tf-mobile_food_segmenter_V1"
    name = "mobile_food_segmenter_V1"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 513, 513, 3), scale=1.)

    if True:
        benchmark(url, dest, onnx_name, opset, imgs, tag='')
        # The conversion works but tensorflow fails with
        # TypeError: 'AutoTrackable' object is not callable

    if True:
        import tensorflow.compat.v2 as tf
        import tensorflow_hub as hub

        m = hub.KerasLayer('https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1')
        inputs = {
            "X": tf.keras.Input(shape=[1, 513, 513, 3], dtype="float32", name="X"),
        }
        outputs = m(inputs)["default"]
        # TypeError: pruned(images) missing required arguments: images
        print(outputs)
        model = tf.keras.Model(inputs, outputs)

        if not os.path.exists(dest):
            os.makedirs(dest)

        # This model is a large model.
        tf2onnx.convert.from_keras(model, opset=13, output_path=onnx_name)


if __name__ == "__main__":
    main()
