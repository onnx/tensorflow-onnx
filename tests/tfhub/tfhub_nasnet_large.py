# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/5?tf-hub-format=compressed"
    dest = "tf-nasnet-large"
    name = "nasnet-large"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 331, 331, 3))

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
