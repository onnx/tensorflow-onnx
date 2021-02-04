# SPDX-License-Identifier: Apache-2.0


"""const dequantize Optimizer.
   if a dequantize op's inputs are const we may be able to fold it through the next op
"""

from .optimizer_base import GraphOptimizerBase
from .const_fold_optimizer import ConstFoldOptimizer

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class ConstDequantizeOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ConstDequantizeOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._fold_node(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _fold_node(self, node, graph):
        """ if a dequantize op's inputs are const and it is fed into a tensor reshaping op, we can apply the op
            directly to the quantized inputs.  Returns True if the graph is changed.
        """
        if node.type not in ["Transpose", "Reshape", "Unsqueeze"]:
            return False
        dequant_node = node.inputs[0]
        if dequant_node.type != "DequantizeLinear":
            return False
        if len(graph.find_output_consumers(dequant_node.output[0])) > 1:
            return False
        if not self._all_inputs_are_const(node.inputs[1:]) or self._is_graph_output(node, graph):
            return False
        if not self._all_inputs_are_const(dequant_node.inputs):
            return False
        graph.replace_input(node, node.input[0], dequant_node.input[0], 0)
        const_outputs = ConstFoldOptimizer.compute_const_folding(node, graph)
        graph.replace_all_inputs(node.output[0], dequant_node.output[0])
        graph.remove_node(node.name)
        dequant_const = dequant_node.inputs[0]
        if len(graph.find_output_consumers(dequant_const.output[0])) > 1:
            dequant_const = graph.copy_const(dequant_const)
            graph.replace_input(dequant_node, dequant_node.input[0], dequant_const.output[0], 0)
        dequant_const.set_tensor_value(const_outputs[0])
        return True

    @staticmethod
    def _all_inputs_are_const(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _is_graph_output(node, graph):
        node_out_set = set(node.output)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)
