# SPDX-License-Identifier: Apache-2.0
import os
import numpy
import numpy.random as rnd
from collections import OrderedDict
from _tools import generate_text_inputs, benchmark


def main(opset=13):
    url = "https://tfhub.dev/tensorflow/albert_en_xlarge/3?tf-hub-format=compressed"
    dest = "tf-albert-en-xlarge"
    name = "albert-en-xlarge"
    onnx_name = os.path.join(dest, "%s-%d.zip" % (name, opset))

    inputs = generate_text_inputs()
    benchmark(url, dest, onnx_name, opset, inputs, output_name="pooled_output")

    inputs = [OrderedDict([
        ('input_word_ids', numpy.array([rnd.randint(0, 1000) for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1))),
        ('input_mask', numpy.array([rnd.randint(0, 1) for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1))),
        ('input_type_ids', numpy.array([i//5 for i in range(0, 128)], dtype=numpy.int32).reshape((1, -1)))
    ]) for i in range(0, 10)]
   
    benchmark(url, dest, onnx_name, opset, inputs, output_name="pooled_output")


if __name__ == "__main__":
    main()
