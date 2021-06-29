# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, measure_time, download_model, convert_model, benchmark

url = "https://tfhub.dev/google/movenet/singlepose/thunder/3?tf-hub-format=compressed"
dest = "tf-thunder"
name = "thunder"
opset = 13
onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

imgs = generate_random_images(shape=(1, 256, 256, 3), dtype=numpy.int32)

benchmark(url, dest, onnx_name, opset, imgs,
          signature='serving_default')
