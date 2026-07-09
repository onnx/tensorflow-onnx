# SPDX-License-Identifier: Apache-2.0


"""
common constants
"""

from onnx import helper

TF2ONNX_PACKAGE_NAME = __name__.split('.')[0]

# Built-in supported domains
ONNX_DOMAIN = ""
AI_ONNX_ML_DOMAIN = "ai.onnx.ml"
MICROSOFT_DOMAIN = "com.microsoft"
CONTRIB_OPS_DOMAIN = "ai.onnx.contrib"

# Default opset version for onnx domain.
# The current update policy is that the default should be set to
# the latest released version as of 18 months ago.
# Opset 15 was released in ONNX v1.10.0 (Jul, 2021).
PREFERRED_OPSET = 15

# Default opset for custom ops
TENSORFLOW_OPSET = helper.make_opsetid("ai.onnx.converters.tensorflow", 1)

# Built-in supported opset
AI_ONNX_ML_OPSET = helper.make_opsetid(AI_ONNX_ML_DOMAIN, 2)

# Target for the generated onnx graph. It possible targets:
# onnx-1.1 = onnx at v1.1 (winml in rs4 is based on this)
# caffe2 = include some workarounds for caffe2 and winml
TARGET_RS4 = "rs4"
TARGET_RS5 = "rs5"
TARGET_RS6 = "rs6"
TARGET_CAFFE2 = "caffe2"
TARGET_TENSORRT = "tensorrt"
TARGET_CHANNELS_LAST = "nhwc"
TARGET_CHANNELS_FIRST = "nchw"
POSSIBLE_TARGETS = [TARGET_RS4, TARGET_RS5, TARGET_RS6, TARGET_CAFFE2, TARGET_TENSORRT, TARGET_CHANNELS_LAST]
DEFAULT_TARGET = []

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
NDHWC_TO_NCDHW = [0, 4, 1, 2, 3]
NCDHW_TO_NDHWC = [0, 2, 3, 4, 1]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]

# Environment variables
ENV_TF2ONNX_DEBUG_MODE = "TF2ONNX_DEBUG_MODE"
ENV_TF2ONNX_CATCH_ERRORS = "TF2ONNX_CATCH_ERRORS"

# Mapping opset to IR version.
# Derived from the ONNX release table (helper.VERSION_TABLE rows are
# (onnx_version, ir_version, opset, ai.onnx.ml, ai.onnx.training)) so that newer
# ONNX releases are picked up automatically instead of hand-editing this map.
# The map is naturally capped by the installed onnx package's known opsets.
# Refer to https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
OPSET_TO_IR_VERSION = {opset: ir_version for _, ir_version, opset, *_ in helper.VERSION_TABLE}
# Low-opset overrides for 1-8; this whole range must stay explicit because:
#   * opsets 2, 3 and 4 have no row in VERSION_TABLE, so they are backfilled here
#     (dropping them would make graph.py reject conversions at those opsets).
#   * opsets 7 and 8 shipped with IR3, but tf2onnx emits PlaceholderWithDefault
#     which requires IR4, so they are pinned to 4 rather than the table's value.
OPSET_TO_IR_VERSION.update({1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4})
