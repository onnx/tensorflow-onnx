# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Back_To_Back Optimizer.
   Collapse consecutive nodes into 1 node if possible.
"""

from __future__ import unicode_literals

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ

_func_map = {}


def _register_func(op_type):
    def _internal_fun(func):
        _func_map[op_type] = func
        return func

    return _internal_fun


class BackToBackOptimizer(GraphOptimizerBase):
    """Remove back-to-back nodes e.g. 'Cast'
    """

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(BackToBackOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, g):
        for optype, handler in _func_map.items():

            # nodes to optimize
            nodes = [n for n in g.get_nodes()
                     if n.type == optype]
            # and not set(n.output) & set(g.outputs)]

            # find consumer list for each cast node
            has_dependencies = set()
            consumer_node_ids = {n.output[0]: [] for n in nodes}
            for n in nodes:
                if n.input[0] in consumer_node_ids:
                    consumer_node_ids[n.input[0]].extend([n])
                    has_dependencies.add(n.output[0])

            # nodes with no dependencies
            q = list(set(consumer_node_ids.keys()) - has_dependencies)
            while q:
                nodeid = q[0]
                q.remove(nodeid)
                node = g.get_node_by_output(nodeid, False)
                downstream_nodes = consumer_node_ids[nodeid]

                if len(downstream_nodes) > 0:
                    # these are back-to-back nodes
                    all_consumers = g.find_output_consumers(nodeid)
                    if len(all_consumers) != len(downstream_nodes):
                        # if first node is used elsewhere, skip
                        continue
                    if set(node.output) & set(g.outputs):
                        # if this node is part of graph outputs, skip
                        continue
                    # update downstream nodes, delete this one
                    for node2 in downstream_nodes:
                        handler(g, node, node2)
                        # add node2 to q, in case it has downstream nodes
                        q.append(node2.output[0])
                    g.remove_node(node.name)

        return g

    @staticmethod
    @_register_func("Cast")
    def _fold_cast(g, node1, node2):
        # TODO: check for cast safety
        node2.input[0] = node1.input[0]

    @staticmethod
    @_register_func("Transpose")
    def _fold_cast2(g, node1, node2):
        node2.input[0] = node1.input[0]
        t1 = list(node1.get_attr("perm").ints)
        t2 = list(node2.get_attr("perm").ints)
        new_perm = [t1[i] for i in t2]

        # check if node2 can be removed. otherwise only update
        if new_perm == list(range(len(t2))) \
                and not set(node2.output) & set(g.outputs):
            # both nodes can be deleted
            # node1 will be removed by caller.so only remove node2 here
            node2_consumers = g.find_output_consumers(node2.output[0])
            for consumer in node2_consumers:
                consumer.input[0] = node1.input[0]
            g.remove_node(node2.name)
        else:
            node2.set_attr("perm", [t1[i] for i in t2])
