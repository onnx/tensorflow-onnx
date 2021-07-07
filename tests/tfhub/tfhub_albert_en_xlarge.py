# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/tensorflow/albert_en_xlarge/3?tf-hub-format=compressed"
    dest = "tf-albert-en-xlarge"
    name = "albert-en-xlarge"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 256, 256, 3), dtype=numpy.int32)

    benchmark(url, dest, onnx_name, opset, imgs,
              signature='serving_default')


if __name__ == "__main__":
    main()
