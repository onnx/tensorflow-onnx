# SPDX-License-Identifier: Apache-2.0


"""
tfl_controlflow
"""

import copy
import numpy as np
from onnx.onnx_pb import TensorProto

from tf2onnx.handler import tfl_op
from tf2onnx import utils
from tf2onnx.tf_loader import find_function
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.onnx_opset.controlflow import parameter_binding, inline_subgraph


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tfl_op(["TFL_WHILE"])
class TflWhile:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["cond"] = node.attr["cond_subgraph_index"]
        del node.attr["cond_subgraph_index"]
        node.attr["body"] = node.attr["body_subgraph_index"]
        del node.attr["body_subgraph_index"]

def wire_tfl_while_body(g, loop_node_inputs, output_shapes,
                        output_dtypes, cond_graph, scan_outputs):
    """Wire subgraph graph into main."""

    g = copy.deepcopy(g)
    graph_inputs = g.inputs.copy()

    # onnx will pass in cond as argument
    iter_node = g.make_node("Placeholder", [], name=utils.make_name("iteration_num"),
                            output_count=1, dtypes=[TensorProto.INT64], shapes=[[]])
    cond_node = g.make_node("Placeholder", [], name=utils.make_name("cond"),
                            output_count=1, dtypes=[TensorProto.BOOL], shapes=[[]])
    cond_binding = parameter_binding(cond_graph, g.outputs)

    to_remove = set()
    for idx, scan_output in scan_outputs:
        inp = graph_inputs[idx]

        # Remove consumers of scan input
        stack = [inp]
        while stack:
            node = stack.pop()
            if node not in to_remove:
                to_remove.add(node)
                for out in node.output:
                    stack += g.find_output_consumers(out)

        # Remove scan input from cond graph
        cond_binding = {k: "@@ALLOC" if v == g.outputs[idx] else v for k, v in cond_binding.items()}
        del g.inputs[idx]
        del g.outputs[idx]
        g.outputs.append(scan_output)

    for node in to_remove:
        g.remove_node(node.name)

    # in onnx the body inputs are: index, cond, [loop_vars]
    g.inputs = [iter_node, cond_node] + g.inputs

    # Shapes of iteration and cond are already known
    for p, c in zip(loop_node_inputs[2:], g.input_names[2:]):
        shape = p.output_shapes[0]
        g.set_shape(c, shape)

    cond_outputs = inline_subgraph(g, cond_graph, "cond__", cond_binding)

    g.outputs = [cond_outputs[0]] + g.outputs
    return g

@tfl_op(["TFL_IF"], tf_op="If")
class TflIfOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["then_branch"] = node.attr["then_subgraph_index"]
        del node.attr["then_subgraph_index"]
        node.attr["else_branch"] = node.attr["else_subgraph_index"]
        del node.attr["else_subgraph_index"]
