# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
    dest = "tf-yamnet-tf"
    name = "yamnet"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(16000, ), dtype=numpy.float32, scale=0.)

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
