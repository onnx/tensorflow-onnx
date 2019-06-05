# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
    This is the optimizer for merging Consts with same Value.
    It is important to do this pior to the Duplicate node removal since
    unifying Consts as inputs to duplicated nodes will be the key
    to grouping such nodes
"""

from collections import defaultdict, namedtuple

from .optimizer_base import GraphOptimizerBase
import re

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

_KeyToGroupNodes = namedtuple("key", "type input")


class MergeConstsOptimizer(GraphOptimizerBase):
    """Remove duplicate nodes.
    """

    def __init__(self):
        super(MergeConstsOptimizer, self).__init__()

    def _optimize(self, graph):
        if graph._opset < 9:
            return graph  # no optimization for IR < 4
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        self._merge_consts(graph)
        return graph

    def _merge_consts(self, graph):
        # Find all Const nodes
        # group by Value
        groups = defaultdict(list)
        for node in graph.get_nodes():
            if node.type == 'Const':
                key = MergeConstsOptimizer._get_key(node.get_attr('value').t)
                groups[key].append(node)

        for k, v in groups.items():
            first = v[0]
            for i in range(1, len(v)):
                consumers = graph.find_output_consumers(v[i].output[0])
                for con in consumers:
                    graph.replace_input(con, v[i].output[0], first.output[0])
                graph.remove_node(v[i].name)
        return graph

    @staticmethod
    def _get_key(t):
        return '{}-{}-{}'.format(t.dims, t.data_type, t.raw_data)
