# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/humpback_whale/1?tf-hub-format=compressed"
    dest = "tf-humpback-whale"
    name = "humpback-whale"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 331, 331, 3))

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
