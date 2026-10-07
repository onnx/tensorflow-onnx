# SPDX-License-Identifier: Apache-2.0

"""
tf2onnx.rewriter.gelu_rewriter - rewrite tensorflow subgraph to onnx gelu op
"""

import numpy as np

from tf2onnx.graph_matcher import GraphMatcher, OpTypePattern


def rewrite_gelu(g, ops):
    """Rewrite GeLU subgraph to GeLU node."""

    if g.opset < 20:
        ... # TODO: Allow opset 20

    # Note: Matches `erfc` based implementation used by recent versions of TensorFlow
    # TODO: Support older versions of TensorFlow?
    pattern = OpTypePattern("Mul", name="gelu", inputs=[
        OpTypePattern("Mul", name="half_x", inputs=[
            OpTypePattern("Const", name="half"),
            OpTypePattern("*", name="x_outer"),
        ]),
        OpTypePattern("Erfc", name="erfc", inputs=[
            OpTypePattern("Mul", name="neg_x_scaled", inputs=[
                OpTypePattern("Neg", name="neg_x", inputs=[
                    OpTypePattern("*", name="x_inner"),
                ]),
                OpTypePattern("Const", name="scale"),
            ]),
        ]),
    ])
    matcher = GraphMatcher(pattern, allow_reorder=True)
    matches = list(matcher.match_ops(ops))

    for match in matches:
        # Match identified pattern against (0.5 * x * erfc(-x * 1/sqrt(2)))
        maybe_half = match.get_op("half").get_tensor_value()
        if not np.isclose(maybe_half, np.float32(0.5)):
            continue
        scale = match.get_op("scale").get_tensor_value()
        if not np.isclose(scale, np.float32(1 / 2 ** 0.5)):
            continue
        if match.get_op("x_outer") is not match.get_op("x_inner"):
            continue

        # Replace matched GeLU expression with GeLU node
        gelu_expr = match.get_op("gelu")
        x_name = match.get_tensor("x_outer")
        gelu_node = g.make_node(
            "Gelu",
            inputs=[x_name],
            # attr={"approximate": "none"},
            shapes=[g.get_shape(gelu_expr.output[0])],
            dtypes=[g.get_dtype(gelu_expr.output[0])],
            op_name_scope=gelu_expr.name,
        )
        g.replace_all_inputs(gelu_expr.output[0], gelu_node.output[0], ops=ops)
        g.safe_remove_nodes([
            match.get_op("gelu"),
            match.get_op("erfc"),
            match.get_op("neg_x_scaled"),
            match.get_op("neg_x"),
        ])

    return ops
