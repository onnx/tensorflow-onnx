# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/deepmind/enformer/1?tf-hub-format=compressed"
    dest = "tf-enformer"
    name = "enformer"
    onnx_name = os.path.join(dest, "%s-%d.zip" % (name, opset))

    imgs = generate_random_images(shape=(1, 224, 224, 3))

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
