"""Resize Optimizer.
    Replace resize operations with all ones in scale with Identity nodes
"""

from __future__ import unicode_literals
from onnx import helper

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class UpsampleOptimizer(GraphOptimizerBase):
    """Upsample Optimizer."""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(UpsampleOptimizer, self).__init__()
        self._g = None

    def _optimize(self, graph):
        return self._apply_optimization(
            graph,
            self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        self._g = graph
        # replace upsample node with all ones in scale with identity node
        for n in self._g.get_nodes():
            if n.type == "Upsample":
                # upsample in opset <=8 has scales in attributes
                if self._g.opset <= 8:
                    scales = n.get_attr_value("scales")
                    if scales and all([float(s) == 1. for s in scales]):
                        n.type = "Identity"
                        self.logger.debug("replacing " + n.name +
                                          " with Identity operation ")
                # upsample in opset > 8 has scales in input[1]
        return self._g
