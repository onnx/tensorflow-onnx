# SPDX-License-Identifier: Apache-2.0
import os

import numpy
from _tools import benchmark, generate_random_images


def main(opset=13):
    url = "https://tfhub.dev/tensorflow/tutorials/spam-detection/1?tf-hub-format=compressed"
    dest = "tf-spam-detection"
    name = "spam-detection"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images((1, 20), dtype=numpy.int32)

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
