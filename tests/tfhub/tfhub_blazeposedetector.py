# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/mediapipe/tfjs-model/blazeposedetector/1/default/1?tfjs-format=compressed"
    dest = "tf-blazeposedetector"
    name = "blazeposedetector"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 513, 513, 3), scale=1.)

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
