# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Graph Optimizer Base"""

from __future__ import unicode_literals

from tf2onnx import logging, utils


class GraphOptimizerBase(object):
    """optimizer graph to improve performance
    """

    def __init__(self):
        self._logger = logging.getLogger('.'.join(__name__.split('.')[:-1] + [self.__class__.__name__]))

    @property
    def logger(self):
        return self._logger

    @property
    def is_debug_mode(self):
        return utils.is_debug_mode()

    def optimize(self, graph):
        original_node_statistics = graph.dump_node_statistics()
        graph = self._optimize(graph)
        graph.delete_unused_nodes(graph.outputs)
        node_statistics = graph.dump_node_statistics()
        self._print_stat_diff(original_node_statistics, node_statistics)
        return graph

    def _optimize(self, graph):
        raise NotImplementedError

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
        self.logger.info("the optimization gain is %s", res)
