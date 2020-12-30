from tf2onnx.handler import tfl_op
from tf2onnx import constants, utils
import numpy as np

guess_list = """ABS
CEIL
COS
ELU
EQUAL
EXP
FLOOR
FLOOR_DIV
FLOOR_MOD
GREATER
GREATER_EQUAL
LESS
LESS_EQUAL
LOG
LOG_SOFTMAX
LOGICAL_AND
LOGICAL_NOT
LOGICAL_OR
MATRIX_DIAG
MATRIX_SET_DIAG
MAXIMUM
MINIMUM
NEG
NOT_EQUAL
POW
RANK
RELU
RELU6
ROUND
RSQRT
SELECT
SELECT_V2
SIN
SQRT
SQUARE
SQUARED_DIFFERENCE
TANH
WHERE
ZEROS_LIKE
"""
guess_list += """FILL
GATHER_ND
PAD
REVERSE_V2
SCATTER_ND
SEGMENT_SUM
SHAPE
SLICE
SQUEEZE
TILE
TRANSPOSE
UNPACK
"""
guess_list += """ADD_N
ONE_HOT
DEPTH_TO_SPACE
ARG_MIN
ARG_MAX
"""
guess_list += """
NON_MAX_SUPPRESSION_V5
RESIZE_NEAREST_NEIGHBOR
LEAKY_RELU
STRIDED_SLICE
MEAN
SUM
MIRROR_PAD
RESIZE_BILINEAR
REVERSE_SEQUENCE
SPARSE_TO_DENSE
CUMSUM
"""
guess_list = guess_list.strip().split("\n")
from tf2onnx.tflite_utils import snake_to_proper_case
direct_tfl_to_tf_map = {"TFL_" + k: snake_to_proper_case(k.lower()) for k in guess_list}
for tfl_op_name, tf_op_name in direct_tfl_to_tf_map.items():
    @tfl_op([tfl_op_name], tf_op=tf_op_name)
    class TflDirectOp:
        @classmethod
        def to_tf(cls, ctx, node, **kwargs):
            pass

