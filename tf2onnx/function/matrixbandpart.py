"""
tf2onnx.tf2onnx - matrixbandpart op conversion
"""
import numpy as np
from onnx import helper, onnx_pb
from tf2onnx import utils
from tf2onnx.utils import make_sure


# pylint: disable=unused-argument,missing-docstring


def matrixbandpart_op(ctx, node, name, args):
    # T output = MatrixBandPart(T input, int num_lower, int num_upper)
    # data-flow: first generate mask matrix and then use element-wise mul op
    input_rank = len(ctx.get_shape(node.input[0]))
    make_sure(input_rank == 2, error_msg="MatrixBandPart op: only rank 2 is supported")
    bandpart = [node.inputs[ind].get_tensor_value()[0] for ind in [1, 2]]
    utils.make_sure(bandpart in [[-1, 0], [0, -1]], "only support Lower/Upper triangular for now")
    # methods to generate mask matrix: if lower triangular is needed, then generate column one by one
    # otherwise row is generated one by one.
    axis, counter_axis, squeeze_axis = (1, 0, 2) if bandpart == [-1, 0] else (0, 1, 1)
    nodes = []
    # 1: subgraph to implement tf.onelike(input[:, 0]),
    # no need to worry about the dtype, because bool type is needed as Xor only support bool
    node_name = utils.make_name("const_zero")
    const_zero = ctx.make_const(name=node_name, np_val=np.array([0]).astype(np.int32))
    first_col_or_row = ctx.make_node(op_type="Gather", inputs=[node.input[0], const_zero.output[0]],
                                     attr={"axis": axis})
    nodes.append(first_col_or_row)
    first_col_or_row_casted = ctx.make_node(op_type="Cast", inputs=first_col_or_row.output,
                                            attr={"to": onnx_pb.TensorProto.BOOL})
    nodes.append(first_col_or_row_casted)
    # line means one col or one row
    zero_line = ctx.make_node(op_type="Xor", inputs=first_col_or_row_casted.output*2)
    nodes.append(zero_line)
    one_line = ctx.make_node(op_type="Not", inputs=zero_line.output)
    nodes.append(one_line)

    # 2: "loop" to generate mask matrix: generate col or row of matrix one by one
    body_ins = [utils.make_onnx_inputs_outputs("trip", onnx_pb.TensorProto.INT64, []),
                utils.make_onnx_inputs_outputs("cond", onnx_pb.TensorProto.BOOL, []),
                utils.make_onnx_inputs_outputs("line", onnx_pb.TensorProto.BOOL, [-1, -1])]
    body_outs = [utils.make_onnx_inputs_outputs("cond_out", onnx_pb.TensorProto.BOOL, []),
                 utils.make_onnx_inputs_outputs("line_out", onnx_pb.TensorProto.BOOL, [-1, -1]),
                 utils.make_onnx_inputs_outputs("res", onnx_pb.TensorProto.BOOL, [-1, -1])]
    # body graph
    node_name = utils.make_name("const_zero_bool")
    const_zero_bool = ctx.make_const(name=node_name, np_val=np.array([[0]]).astype(np.bool))
    slice_node = ctx.make_node(op_type="Slice", inputs=["line"],
                               attr={"axes": [counter_axis], "starts": [0], "ends": [-1]})
    new_line = ctx.make_node(op_type="Concat", inputs=[const_zero_bool.output[0], slice_node.output[0]],
                             outputs=["line_out"], attr={"axis": counter_axis})
    body_nodes = [slice_node.op, new_line.op,
                  utils.make_onnx_identity("cond", "cond_out"),
                  utils.make_onnx_identity("line", "res")]
    body_graph = helper.make_graph(body_nodes, utils.make_name("MatrixBandPart_body"), body_ins, body_outs)
    # initial value of body vars
    shape = ctx.make_node(op_type="Shape", inputs=[node.input[0]])  # dtype of result is int64
    nodes.append(shape)
    node_name = utils.make_name("line_num_index")
    col_or_row_num_index = ctx.make_const(name=node_name, np_val=np.array(axis).astype(np.int32))
    line_num = ctx.make_node(op_type="Gather", inputs=[shape.output[0], col_or_row_num_index.output[0]])
    nodes.append(line_num)
    trip_cnt = line_num.output[0]
    node_name = utils.make_name("true")
    cond = ctx.make_const(name=node_name, np_val=np.array(1).astype(np.bool)).output[0]
    col_init = one_line.output[0]

    loop_node = ctx.make_node(op_type="Loop", inputs=[trip_cnt, cond, col_init], output_count=2,
                              attr={"body": body_graph})
    nodes.append(loop_node)
    # convert generated mask matrix from bool to right shape and data type
    squeeze = ctx.make_node(op_type="Squeeze", inputs=[loop_node.output[1]], attr={"axes": [squeeze_axis]})
    nodes.append(squeeze)
    cast1 = ctx.make_node(op_type="Cast", inputs=squeeze.output, attr={"to": onnx_pb.TensorProto.FLOAT})
    nodes.append(cast1)
    if axis == 1:
        mask_matrix = ctx.make_node(op_type="Transpose", inputs=cast1.output)
        nodes.append(mask_matrix)
    else:
        mask_matrix = squeeze
    cast2 = ctx.make_node(op_type="Cast", inputs=mask_matrix.output,
                          attr={"to": ctx.get_dtype(node.input[0])})
    nodes.append(cast2)
    res = ctx.make_node(op_type="Mul", inputs=[cast2.output[0], node.input[0]],
                        name=node.name, outputs=node.output)
    nodes.append(res)
    return nodes
