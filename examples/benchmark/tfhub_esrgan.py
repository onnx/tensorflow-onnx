# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, measure_time, download_model, convert_model, benchmark

url = "https://tfhub.dev/captain-pool/esrgan-tf2/1?tf-hub-format=compressed"
dest = "tf-esrgan-tf2"
name = "esrgan-tf2"
opset = 13
onnx_name = os.path.join(dest, "esrgan-tf2-%d.onnx" % opset)

imgs = generate_random_images()

benchmark(url, dest, onnx_name, opset, imgs)
