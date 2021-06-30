# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark_tflite


def main(opset=13):
    url = "https://tfhub.dev/google/coral-model/yamnet/classification/coral/1?coral-format=tflite"
    dest = "tf-yamnet"
    name = "yamnet"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 256, 256, 3), dtype=numpy.int32)

    benchmark_tflite(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
