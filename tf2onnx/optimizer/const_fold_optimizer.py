# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""const fold Optimizer.
   if op's inputs are all const then do op computation when building the graph to improve performance
   for example, input of transpose node is const then we can do transpose statically instead of at runtime
"""

import logging

from tf2onnx.optimizer.optimizer_base import GraphOptimizerBase
from tf2onnx import utils

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

_func_map = {}


def _register_func(onnx_op):
    def _internal_fun(func):
        _func_map[onnx_op] = func
        return func
    return _internal_fun


class ConstFoldOptimizer(GraphOptimizerBase):

    def __init__(self, debug=False):
        super(ConstFoldOptimizer, self).__init__("ConstFoldOptimizer", debug)
        self._log = logging.getLogger("tf2onnx.optimizer.%s" % self._name)

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._can_skip(op):
                    continue
                if self._fold_node(op, graph):
                    graph_changed = True
        return graph

    @staticmethod
    def _can_skip(node):
        if node.is_const() or node.is_graph_input():
            return True

        skip_type = ["Identity"]
        if node.type in skip_type:
            return True

        return False

    def _fold_node(self, node, graph):
        """ if node's input are all const and it's not graph's output then it can be fold.
            if node can be fold True will be return indicating that graph is changed
        """
        if self._all_inputs_are_const(node.inputs) and not self._is_graph_output(node, graph):
            process_func = _func_map.get(node.type, self._try_fold)
            return process_func(node, graph)

        return False

    @staticmethod
    def _all_inputs_are_const(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _is_graph_output(node, graph):
        node_out_set = set(node.output)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)

    @staticmethod
    def _replace_node_with_const(node, graph, val):
        const_node = graph.make_const(utils.make_name("const_fold_opt"), val)
        graph.replace_all_inputs(graph.get_nodes(), node.output[0], const_node.output[0])
        graph.remove_node(node.name)

    @staticmethod
    @_register_func(onnx_op="Transpose")
    def _fold_transpose(node, graph):
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        perm_attr = node.get_attr("perm")
        perm = perm_attr.ints if perm_attr else None
        const_val_after_trans = const_val.transpose(perm)
        ConstFoldOptimizer._replace_node_with_const(node, graph, const_val_after_trans)
        return True

    def _try_fold(self, node, graph):
        self._log.warning("need to add function to fold op %s whose op_type is %s", node.name, node.type)
        return False
