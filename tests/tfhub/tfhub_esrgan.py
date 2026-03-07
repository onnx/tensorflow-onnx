# SPDX-License-Identifier: Apache-2.0
import os

from _tools import benchmark, generate_random_images


def main(opset=13):
    url = "https://tfhub.dev/captain-pool/esrgan-tf2/1?tf-hub-format=compressed"
    dest = "tf-esrgan-tf2"
    name = "esrgan-tf2"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images()

    benchmark(url, dest, onnx_name, opset, imgs)


if __name__ == "__main__":
    main()
