# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark_tflite


def main(opset=13):
    url = "https://tfhub.dev/google/coral-model/yamnet/classification/coral/1?coral-format=tflite"
    dest = "tf-yamnet-coral"
    name = "yamnet"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 256, 256, 3), dtype=numpy.int32)

    benchmark_tflite(url, dest, onnx_name, opset, imgs)
    # WARNING - Error loading model into tflite interpreter: Encountered unresolved custom op: edgetpu-custom-op.Node number 14 (edgetpu-custom-op) failed to prepare.
    # WARNING - Could not parse attributes for custom op 'TFL_edgetpu-custom-op': 'utf-8' codec can't decode byte 0xc8 in position 0: invalid continuation byte
    # WARNING - For now, onnxruntime only support float32 type for Gemm rewriter
    # ERROR - Tensorflow op [tower0/network/layer32/final_output1_prequant: TFL_edgetpu-custom-op] is not supported
    # ERROR - Unsupported ops: Counter({'TFL_edgetpu-custom-op': 1})

if __name__ == "__main__":
    main()
