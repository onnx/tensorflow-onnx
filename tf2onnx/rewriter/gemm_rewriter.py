# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewrite - rewrite tensorflow subgraph to onnx gemm op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from onnx import onnx_pb
import logging

# pylint: disable=missing-docstring

def rewrite_gemm(g, ops):
    if g.opset <= 6:
        return ops

    """
    4 Candidate patterns are listed as follow, i.e. pattern0, pattern1, pattern2 and pattern 3
    Where, A,B and C represent the three inputs, alpha and beta represent the two attributes.
    """
    # pattern0: alpha*A*B + beta*C
    pattern0 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('MatMul', name='matmul')
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern1: alpha*A*B + C
    pattern1 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('MatMul', name='matmul'),
                OpTypePattern('Const', name='alpha')
            ]),
            OpTypePattern('*', name='C'),
        ])

    # pattern2: A*B + beta*C
    pattern2 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern3: A*B + C
    pattern3 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('*', name='C'),
        ])

    pattern_list = [pattern0, pattern1, pattern2, pattern3]

    for pattern_id, pattern in enumerate(pattern_list):
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if len(match_results) > 0:
            for match in match_results:
                add_node = match.get_op('add')
                matmul_node = match.get_op("matmul")
                input_c_node = match.get_op("C")

                if g.get_dtype(matmul_node.input[0]) != onnx_pb.TensorProto.FLOAT:
                    logging.warning(u"For now, onnxruntime only support float type for Gemm rewriter")
                    return ops
                else:
                    a_edge_name = matmul_node.input[0]
                    b_edge_name = matmul_node.input[1]
                    c_edge_name = input_c_node.output[0]
                attr = {}

                # For each pattern, we must ensure that alpha and beta are both scalar, or return ops
                if pattern_id == 0:  # pattern 0: alpha*A*B + beta*C
                    alpha = match.get_op("alpha").get_tensor_value()
                    beta = match.get_op("beta").get_tensor_value()
                    if isinstance(alpha, float) or isinstance(alpha, int):
                        alpha = float(alpha)
                    else:
                        return ops
                    if isinstance(beta, float) or isinstance(beta, int):
                        beta = float(beta)
                    else:
                        return ops
                    attr = {"alpha": alpha, "beta": beta}

                if pattern_id == 1:  # pattern1: alpha*A*B + C
                    alpha = match.get_op("alpha").get_tensor_value()
                    if isinstance(alpha, float) or isinstance(alpha, int):
                        alpha = float(alpha)
                    else:
                        return ops
                    attr = {"alpha": alpha}

                if pattern_id == 2:  # pattern2: A*B + beta*C
                    beta = match.get_op("beta").get_tensor_value()
                    if isinstance(beta, float) or isinstance(beta, int):
                        beta = float(beta)
                    else:
                        return ops
                    attr = {"beta": beta}

                # if the pattern is 3, do nothing

                gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                                   attr=attr,
                                   shapes=[g.get_shape(add_node.output[0])],
                                   dtypes=[g.get_dtype(add_node.output[0])])

                ops.append(gemm)
                g.replace_all_inputs(ops, add_node.output[0], gemm.output[0])
                to_delete = [add_node, matmul_node]
                g.safe_remove_nodes(to_delete)

    return ops
