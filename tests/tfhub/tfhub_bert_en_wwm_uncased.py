# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
import numpy
import numpy.random as rnd
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/4?tf-hub-format=compressed"
    dest = "tf-bert-en-wwm-uncased-L-24-H-1024-A-16"
    name = "bert-en-wwm-uncased-L-24-H-1024-A-16"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    inputs = [OrderedDict([
        ('input_word_ids', numpy.array([rnd.randint(0, 1000) for i in range(0, 32)], dtype=numpy.int32).reshape((1, -1))),
        ('input_mask', numpy.array([rnd.randint(0, 1) for i in range(0, 32)], dtype=numpy.int32).reshape((1, -1))),
        ('input_type_ids', numpy.array([i//5 for i in range(0, 32)], dtype=numpy.int32).reshape((1, -1)))
    ]) for i in range(0, 10)]

    benchmark(url, dest, onnx_name, opset, inputs, output_name="pooled_output")


if __name__ == "__main__":
    main()
