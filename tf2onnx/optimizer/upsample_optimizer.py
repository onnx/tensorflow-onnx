"""Resize Optimizer.
    Replace resize operations with all ones in scale with Identity nodes
"""

from __future__ import unicode_literals

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class UpsampleOptimizer(GraphOptimizerBase):
    """Upsample Optimizer."""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(UpsampleOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(
            graph,
            self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        # replace upsample node with all ones in scale with identity node
        for n in graph.get_nodes():
            if n.type == "Upsample":
                scales = n.get_attr_value("scales")
                if all([s == 1 for s in scales]):
                    n.type = "Identity"
                    if len(n.input) > 0:
                        n.input = [n.input[0]]
                    self.logger.debug("replacing " + n.name +
                                      " with Identity operation")
        return graph
