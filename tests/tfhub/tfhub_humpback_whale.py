# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/humpback_whale/1?tf-hub-format=compressed"
    dest = "tf-humpback-whale"
    name = "humpback-whale"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 1024, 1))
    inputs = [dict(waveform=img,
                   context_step_samples=numpy.array(512, dtype=numpy.int64))
              for img in imgs]

    benchmark(url, dest, onnx_name, opset, inputs, optimize=False)


if __name__ == "__main__":
    main()
