# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
import numpy
import numpy.random as rnd
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1?tf-hub-format=compressed"
    dest = "tf-mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT"
    name = "mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT"
    onnx_name = os.path.join(dest, "%s-%d_no_einsum.onnx" % (name, opset))

    inputs = [OrderedDict([
        ('input_word_ids', numpy.array([rnd.randint(0, 1000) for i in range(0, 1024)], dtype=numpy.int32).reshape((1, -1))),
        ('input_mask', numpy.array([rnd.randint(0, 1) for i in range(0, 1024)], dtype=numpy.int32).reshape((1, -1))),
        ('input_type_ids', numpy.array([i//5 for i in range(0, 1024)], dtype=numpy.int32).reshape((1, -1)))
    ]) for i in range(0, 10)]

    benchmark(url, dest, onnx_name, opset, inputs)


if __name__ == "__main__":
    main()
