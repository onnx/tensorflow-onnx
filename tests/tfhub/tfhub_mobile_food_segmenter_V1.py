# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1?tf-hub-format=compressed"
    dest = "tf-mobile_food_segmenter_V1"
    name = "mobile_food_segmenter_V1"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 513, 513, 3), scale=1.)

    benchmark(url, dest, onnx_name, opset, imgs, tag='')


if __name__ == "__main__":
    main()
