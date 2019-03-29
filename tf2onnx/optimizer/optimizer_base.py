# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Graph Optimizer Base"""

from __future__ import unicode_literals
import logging


class GraphOptimizerBase(object):
    """optimizer graph to improve performance
    """

    def __init__(self, name, debug=False):
        self._debug = debug
        self._name = name
        self._log = logging.getLogger("tf2onnx.optimizer.%s" % self._name)

    def optimize(self, graph):
        original_node_statistics = graph.dump_node_statistics()
        graph = self._optimize(graph)
        graph.delete_unused_nodes(graph.outputs)
        graph.topological_sort(graph.get_nodes())
        node_statistics = graph.dump_node_statistics()
        self._print_stat_diff(original_node_statistics, node_statistics)
        return graph

    def _optimize(self, graph):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def log(self):
        return self._log

    @staticmethod
    def _apply_optimization(graph, optimize_func):
        """
        optimize graph
        will also optimize graph of nodes'
        Args:
            graph: the top level graph to be optimized
            optimize_func: function to optimize graph
        """
        graph = optimize_func(graph)
        for node in graph.get_nodes():
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for attr, b_g in body_graphs.items():
                    b_g = optimize_func(b_g)
                    node.set_body_graph_as_attr(attr, b_g)
        return graph

    def _print_stat_diff(self, nodes_original, nodes_after_optimized):
        nodes_after_optimized.subtract(nodes_original)
        res = {}
        for key, value in nodes_after_optimized.items():
            if value != 0:
                res[key] = value
        self.log.info("the optimization gain is %s", res)
