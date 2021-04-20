# SPDX-License-Identifier: Apache-2.0

"""tf2onnx.tflite_rewriters module"""

from tf2onnx.tflite_rewriters.tfl_scan_output_rewriter import rewrite_tfl_scan_outputs
from tf2onnx.tflite_rewriters.tfl_qdq_rewriter import rewrite_tfl_qdq
from tf2onnx.tflite_rewriters.tfl_select_zero_mul_rewriter import rewrite_tfl_select_zero_mul

__all__ = [
    "rewrite_tfl_scan_outputs",
    "rewrite_tfl_qdq",
    "rewrite_tfl_select_zero_mul",
]
