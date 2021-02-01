# SPDX-License-Identifier: Apache-2.0


"""
tfl_tensor
"""

import logging
from tf2onnx.handler import tfl_op

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tfl_op(["TFL_CONCATENATION"], onnx_op="Concat")
class TflConcatenation:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_SPLIT"], tf_op="Split")
class TflSplit:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr['num_split'] = node.attr['num_splits']
        del node.attr['num_splits']

@tfl_op(["TFL_SPLIT_V"], tf_op="SplitV")
class TflSplitV:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr['num_split'] = node.attr['num_splits']
        del node.attr['num_splits']

@tfl_op(["TFL_GATHER"], onnx_op="Gather")
class TflGather:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_RESHAPE"], tf_op="Reshape")
class TflReshape:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        if 'new_shape' in node.attr:
            del node.attr['new_shape']
        #utils.make_sure('new_shape' not in node.attr, "new_shape attr not yet supported for reshape (use input)")

@tfl_op(["TFL_CAST"], tf_op="Cast")
class TflCast:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        dst = ctx.get_dtype(node.output[0])
        if "out_data_type" in node.attr:
            del node.attr["out_data_type"]
            del node.attr["in_data_type"]
        node.set_attr("to", dst)

@tfl_op(["TFL_PACK"], tf_op="Pack")
class TFlPackOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["N"] = node.attr["values_count"]
        del node.attr["values_count"]

@tfl_op(["TFL_PADV2"], tf_op="PadV2")
class TflPadV2Op:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_UNIQUE"], tf_op="Unique")
class TFlUniqueOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["out_idx"] = node.attr["idx_out_type"]
        del node.attr["idx_out_type"]

@tfl_op(["TFL_TOPK_V2"], tf_op="TopKV2")
class TFlTopKV2Op:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("sorted", 1)
