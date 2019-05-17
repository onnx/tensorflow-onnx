# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
math
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb
from tf2onnx import constants, utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import common


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

@tf_op(["Add", "Div", "Mul", "Sub"])
class BroadcastOp(common.BroadcastOp):
    pass


@tf_op(["RealDiv", "TruncateDiv"], onnx_op="Div")
class RealDiv(common.BroadcastOp):
    pass


@tf_op(["Abs", "Ceil", "Elu", "Exp", "Floor", "LeakyRelu", "Log", "LogSoftmax", "Neg", "Relu", "Sigmoid", "Sqrt",
        "Tanh", "Softplus", "Softsign", "Reciprocal"])
class DirectOp:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        pass


@tf_op(["Acos", "Asin", "Atan", "Cos", "Sin", "Tan"])
class TrigOpSinceOpset7:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass


@tf_op(["Acosh", "Asinh", "Atanh", "Cosh", "Sinh"])
class TrigOpSinceOpset9:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass


@tf_op("Minimum", onnx_op="Min")
@tf_op("Maximum", onnx_op="Max")
class MinMaxOp:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # tensorflow minimum/maximum does support broadcast, onnx < opset 8 does not.
        # handle this by doing something like:
        # y = min(x1, add(x2, sub(x1, x1))), where x1, x2 are the inputs and x2 is a scalar
        # this will create a tensor of zeros of the shape of x1, adds x2 to it (which broadcasts) and use that for min.
        # support more dtype
        supported_dtypes = [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE
        ]
        target_dtype = onnx_pb.TensorProto.FLOAT
        need_cast = False
        for inp in node.input:
            dtype = ctx.get_dtype(inp)
            utils.make_sure(dtype is not None, "dtype of {} is None".format(inp))
            if dtype not in supported_dtypes:
                inp_cast = ctx.insert_new_node_on_input(node, "Cast", inp, to=target_dtype)
                ctx.copy_shape(inp, inp_cast.output[0])
                ctx.set_dtype(inp_cast.output[0], target_dtype)
                need_cast = True
        if need_cast:
            origin_dtype = ctx.get_dtype(node.output[0])
            utils.make_sure(origin_dtype is not None, "dtype of {} is None".format(node.output[0]))
            ctx.set_dtype(node.output[0], target_dtype)
            cast_name = utils.make_name(node.name)
            cast_node = ctx.insert_new_node_on_output("Cast", node.output[0], name=cast_name, to=origin_dtype)
            ctx.set_dtype(cast_node.output[0], origin_dtype)
            ctx.copy_shape(node.output[0], cast_node.output[0])
            to_replace = [n for n in ctx.get_nodes() if n != cast_node]
            ctx.replace_all_inputs(to_replace, node.output[0], cast_node.output[0])

        shapeo = ctx.get_shape(node.output[0])
        needs_broadcast_op = []
        has_correct_shape = []
        if ctx.opset < 8:
            for i, input_name in enumerate(node.input):
                if ctx.get_shape(input_name) != shapeo:
                    needs_broadcast_op.append(i)
                else:
                    has_correct_shape.append(input_name)
        if needs_broadcast_op:
            has_correct_shape = has_correct_shape[0]
            for i in needs_broadcast_op:
                input_node = node.inputs[i]
                # get a tensor with zeros (since there is no Fill op as of opset8)
                sub_node = ctx.make_node("Sub", [has_correct_shape, has_correct_shape],
                                         op_name_scope=input_node.name)
                # use add as 'broadcast' op
                add_node = ctx.make_node("Add", [input_node.output[0], sub_node.output[0]],
                                         op_name_scope=input_node.name)
                node.input[i] = add_node.output[0]


@tf_op("Softmax")
class Softmax:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = Softmax(T logits). The axis softmax would be performed on is always on -1.
        # T output = Softmax(T input, @int axis). Default axis is 1.
        logits_rank = len(ctx.get_shape(node.input[0]))
        node.set_attr("axis", logits_rank - 1)


@tf_op("Square")
class Square:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        node.type = "Mul"
        node.input.append(node.input[0])


@tf_op("Relu6")
class Relu6:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # relu6 = min(max(features, 0), 6)
        node.type = "Relu"
        clip_name = utils.make_name(node.name)
        clip_node = ctx.insert_new_node_on_output("Clip", node.output[0], name=clip_name, min=0.0, max=6.0)
        ctx.copy_shape(node.output[0], clip_node.output[0])


@tf_op("Rsqrt")
class Rsqrt:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        node.type = "Sqrt"
        op_name = utils.make_name(node.name)
        reciprocal = ctx.insert_new_node_on_output("Reciprocal", node.output[0], name=op_name)
        ctx.copy_shape(node.output[0], reciprocal.output[0])


@tf_op("SquaredDifference")
class SquaredDifference:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        node.type = "Sub"
        op_name = utils.make_name(node.name)
        mul = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
        mul.input.append(node.output[0])


@tf_op("Sign")
class Sign:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        """Sign op."""
        # T sign = Sign(T Input)
        node_dtype = ctx.get_dtype(node.output[0])
        utils.make_sure(node_dtype, "Dtype of {} is None".format(node.name))
        if node_dtype in [onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128]:
            raise ValueError("dtype " + str(node_dtype) + " is not supported in onnx for now")
        zero_name = utils.make_name("{}_zero".format(node.name))
        ctx.make_const(zero_name, np.array(0, dtype=np.float32))
        if node_dtype not in [onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.DOUBLE]:
            cast_node_0 = ctx.make_node("Cast", [node.input[0]], {"to": onnx_pb.TensorProto.FLOAT})
            greater_node = ctx.make_node("Greater", [cast_node_0.output[0], zero_name])
            less_node = ctx.make_node("Less", [cast_node_0.output[0], zero_name])
        else:
            greater_node = ctx.make_node("Greater", [node.input[0], zero_name])
            less_node = ctx.make_node("Less", [node.input[0], zero_name])
        cast_node_1 = ctx.make_node("Cast", [greater_node.output[0]], {"to": node_dtype})
        cast_node_2 = ctx.make_node("Cast", [less_node.output[0]], {"to": node_dtype})

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Sub", [cast_node_1.output[0], cast_node_2.output[0]], outputs=[node.output[0]],
                      shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        node_dtype = ctx.get_dtype(node.output[0])
        utils.make_sure(node_dtype, "dtype of {} is None".format(node.name))
        if node_dtype in [onnx_pb.TensorProto.BOOL, onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128]:
            raise ValueError("dtype " + str(node_dtype) + " is not supported in onnx for now")


@tf_op("Pow")
class Pow:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        if ctx.is_target(constants.TARGET_CAFFE2):
            # workaround a bug in caffe2 pre Feb2018, pow(a, b) becomes np.exp(np.log(a) * b)
            node.type = "Log"
            b = node.input[1]
            ctx.remove_input(node, node.input[1])
            op_name = utils.make_name(node.name)
            mul_op = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
            mul_op.input.append(b)
            op_name = utils.make_name(node.name)
            exp_op = ctx.insert_new_node_on_output("Exp", mul_op.output[0], name=op_name)
            ctx.copy_shape(node.output[0], exp_op.output[0])
            BroadcastOp.version_4(ctx, mul_op, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass


@tf_op("LRN")
class LRN:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # FIXME: numerical results are not correct
        # ONNX: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
        # TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
        #     output = input / (bias + alpha * sqr_sum) ** beta
        depth_radius = node.get_attr("depth_radius")
        if depth_radius:
            size = depth_radius.i
        else:
            size = 5
        node.set_attr("size", size)


@tf_op(["MatMul", "BatchMatMul"])
class MatMul:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # tensorflow allows transpose and conjugated. If found, insert the required transpose.
        # We could use Gemm as well but tensorflow does not pass bias in matmul.
        node.type = "MatMul"

        attrs = ["transpose_a", "transpose_b", "adjoint_a", "adjoint_b", "adj_x", "adj_y"]
        attrs_val = [node.get_attr(attr) for attr in attrs]
        attrs_val = [0 if val is None else val.i for val in attrs_val]

        dtype = ctx.get_dtype(node.output[0])
        if any(attrs_val[2:]):
            # conjugation operation on complex data not supported in onnx for now
            # so if it's complex than raise exception
            if dtype not in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
                raise ValueError("dtype " + dtype + " is not supported in onnx matmul for now")

        transpose_a = (attrs_val[0] + attrs_val[2] + attrs_val[4]) % 2
        transpose_b = (attrs_val[1] + attrs_val[3] + attrs_val[5]) % 2

        if transpose_a != 0:
            shape = ctx.get_shape(node.input[0])
            if shape:
                perm = list(range(0, len(shape)))
                tmp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = tmp
                ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=perm)

        if transpose_b != 0:
            shape = ctx.get_shape(node.input[1])
            if shape:
                perm = list(range(0, len(shape)))
                tmp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = tmp
                ctx.insert_new_node_on_input(node, "Transpose", node.input[1], perm=perm)

        unsupported = ["a_is_sparse", "b_is_sparse"]
        for i in unsupported:
            val = node.get_attr(i)
            if val is not None and val.i != 0:
                raise ValueError(node.type + " attribute " + i + " is not supported")


@tf_op("Erf")
class Erf:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        """Error function."""
        # constant names
        a1 = "erf_a1"
        a2 = "erf_a2"
        a3 = "erf_a3"
        a4 = "erf_a4"
        a5 = "erf_a5"
        p = "erf_p"
        one = "erf_one"
        null = "erf_null"

        n = node.name
        output_name = node.output[0]
        erf_a1_node = ctx.get_node_by_output("erf_a1")
        if erf_a1_node is None:
            # insert the constants for erf once
            ctx.make_const(a1, np.array(0.254829592, dtype=np.float32))
            ctx.make_const(a2, np.array(-0.284496736, dtype=np.float32))
            ctx.make_const(a3, np.array(1.421413741, dtype=np.float32))
            ctx.make_const(a4, np.array(-1.453152027, dtype=np.float32))
            ctx.make_const(a5, np.array(1.061405429, dtype=np.float32))
            ctx.make_const(p, np.array(0.3275911, dtype=np.float32))
            ctx.make_const(one, np.array(1., dtype=np.float32))
            ctx.make_const(null, np.array(0., dtype=np.float32))

        x = node.input[0]

        # erf(x):
        #  sign = 1 if x >= 0 else -1
        #  x = abs(x)
        #  # A&S formula 7.1.26
        #  t = 1.0 / (1.0 + p * x)
        #  y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) *  t * math.exp(-x * x)
        #  return sign * y  # erf(-x) = -erf(x)

        x_node = ctx.make_node("Abs", [x], op_name_scope=node.name, name="x")
        negx_node = ctx.make_node("Sub", [null, x], op_name_scope=node.name, name="negx")
        is_positive_node = ctx.make_node("Greater", [x, null], op_name_scope=node.name, name="isPositive")
        is_positive_value_node = ctx.make_node("Cast", is_positive_node.output, op_name_scope=node.name,
                                               name="isPositiveValue", attr={"to": onnx_pb.TensorProto.FLOAT})
        is_neg_node = ctx.make_node("Less", [x, null], op_name_scope=node.name, name="isNeg")
        ig_neg_value_node = ctx.make_node("Cast", is_neg_node.output, op_name_scope=node.name, name="isNegValue",
                                          attr={"to": onnx_pb.TensorProto.FLOAT})
        sign0_node = ctx.make_node("Sub", [is_positive_value_node.output[0], ig_neg_value_node.output[0]],
                                   op_name_scope=node.name, name="sign0")
        sign_add_one_node = ctx.make_node("Add", [sign0_node.output[0], one], op_name_scope=node.name,
                                          name="signAddOne")
        non_zero_node = ctx.make_node("Abs", sign0_node.output, op_name_scope=node.name, name="nonZero")
        sign_node = ctx.make_node("Sub", [sign_add_one_node.output[0], non_zero_node.output[0]],
                                  op_name_scope=node.name, name="sign")
        num_4_node = ctx.make_node("Mul", [x_node.output[0], p], op_name_scope=node.name, name="4")
        num_5_node = ctx.make_node("Add", [num_4_node.output[0], one], op_name_scope=node.name, name="5")
        t_node = ctx.make_node("Div", [one, num_5_node.output[0]], op_name_scope=node.name, name="t")
        xsq_node = ctx.make_node("Mul", [x, negx_node.output[0]], op_name_scope=node.name, name="xsq")
        num_6_node = ctx.make_node("Exp", xsq_node.output, op_name_scope=node.name, name="6")
        num_7_node = ctx.make_node("Mul", [num_6_node.output[0], t_node.output[0]], op_name_scope=node.name, name="7")
        num_8_node = ctx.make_node("Mul", [t_node.output[0], a5], op_name_scope=node.name, name="8")
        num_9_node = ctx.make_node("Add", [num_8_node.output[0], a4], op_name_scope=node.name, name="9")
        num_10_node = ctx.make_node("Mul", [num_9_node.output[0], t_node.output[0]], op_name_scope=node.name, name="10")
        num_11_node = ctx.make_node("Add", [num_10_node.output[0], a3], op_name_scope=node.name, name="11")
        num_12_node = ctx.make_node("Mul", [num_11_node.output[0], t_node.output[0]], op_name_scope=node.name,
                                    name="12")
        num_13_node = ctx.make_node("Add", [num_12_node.output[0], a2], op_name_scope=node.name, name="13")
        num_14_node = ctx.make_node("Mul", [num_13_node.output[0], t_node.output[0]], op_name_scope=node.name,
                                    name="14")
        num_15_node = ctx.make_node("Add", [num_14_node.output[0], a1], op_name_scope=node.name, name="15")
        num_16_node = ctx.make_node("Mul", [num_15_node.output[0], num_7_node.output[0]], op_name_scope=node.name,
                                    name="16")
        num_17_node = ctx.make_node("Sub", [one, num_16_node.output[0]], op_name_scope=node.name, name="17")

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Mul", [num_17_node.output[0], sign_node.output[0]], outputs=[output_name], name=n,
                      shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass


@tf_op("FloorDiv")
class FloorDiv:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        # T output = FloorDiv(T x, T y)
        node.type = "Div"
        dtype = ctx.get_dtype(node.input[0])
        if dtype in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
            new_node_name = utils.make_name("floor_div_res")
            floor_res = ctx.insert_new_node_on_output(op_type="Floor", output_name=node.output[0],
                                                      name=new_node_name)
            ctx.copy_dtype(node.output[0], floor_res.output[0])
            ctx.copy_shape(node.output[0], floor_res.output[0])


@tf_op("FloorMod")
class FloorMod:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = FloorMod(T x, T y)
        div = ctx.make_node(op_type="Div", inputs=node.input)
        dtype = ctx.get_dtype(node.input[0])
        if dtype in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
            div = ctx.make_node(op_type="Floor", inputs=div.output)

        mul = ctx.make_node(op_type="Mul", inputs=[div.output[0], node.input[1]])
        # res node will take over shape&dtype&output connection info of original "node"
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Sub", inputs=[node.input[0], mul.output[0]],
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
