# SPDX-License-Identifier: Apache-2.0

"""tf2onnx.optimizer module"""

import copy
from collections import OrderedDict

from .. import logging
from .back_to_back_optimizer import BackToBackOptimizer
from .const_dequantize_optimizer import ConstDequantizeOptimizer
from .const_fold_optimizer import ConstFoldOptimizer
from .einsum_optimizer import EinsumOptimizer
from .global_pool_optimizer import GlobalPoolOptimizer
from .identity_optimizer import IdentityOptimizer
from .loop_optimizer import LoopOptimizer
from .merge_duplicated_nodes_optimizer import MergeDuplicatedNodesOptimizer
from .q_dq_optimizer import QDQOptimizer
from .reshape_optimizer import ReshapeOptimizer
from .transpose_optimizer import TransposeOptimizer
from .upsample_optimizer import UpsampleOptimizer

# optimizer sequence need to be considered carefully
_optimizers = OrderedDict([
    ("optimize_transpose", TransposeOptimizer),
    ("remove_redundant_upsample", UpsampleOptimizer),
    ("fold_constants", ConstFoldOptimizer),
    ("const_dequantize_optimizer", ConstDequantizeOptimizer),
    ("loop_optimizer", LoopOptimizer),
    # merge_duplication should be used after optimize_transpose
    # for optimize_transpose may have some trans nodes that can be merge
    ("merge_duplication", MergeDuplicatedNodesOptimizer),
    ("reshape_optimizer", ReshapeOptimizer),
    ("global_pool_optimizer", GlobalPoolOptimizer),
    ("q_dq_optimizer", QDQOptimizer),
    ("remove_identity", IdentityOptimizer),
    ("remove_back_to_back", BackToBackOptimizer),
    ("einsum_optimizer", EinsumOptimizer),
])


def _get_optimizers():
    return _optimizers


def optimize_graph(graph, catch_errors=True, optimizers=None):
    """ Optimize graph, return optimized graph. Catch errors and restore old graph if catch_errors is True"""
    logger = logging.getLogger(__name__)
    logger.info("Optimizing ONNX model")

    before = graph.dump_node_statistics()
    opts = _get_optimizers() if optimizers is None else optimizers
    continue_flag = True
    iteration = 0
    while continue_flag:
        continue_flag = False
        for name, factory in opts.items():
            logger.verbose("Apply %s", name)
            if catch_errors:
                try:
                    current = copy.deepcopy(graph)
                    opt = factory()
                    graph = opt.optimize(current, iteration) or graph
                    continue_flag = continue_flag or opt.graph_been_opt
                except Exception:  # pylint: disable=broad-except
                    # if current optimizer fails, continue with other optimizers
                    logger.warning("Failed to apply %s", name, exc_info=1)
            else:
                opt = factory()
                graph = opt.optimize(graph, iteration)
                continue_flag = continue_flag or opt.graph_been_opt
        iteration += 1

    try:
        graph.topological_sort(graph.get_nodes())
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed topological_sort", exc_info=1)

    after = graph.dump_node_statistics()
    diff = copy.deepcopy(after)
    diff.subtract(before)
    diff = ["{} {} ({}->{})".format(k, str(v) if v < 0 else '+' + str(v), before.get(k, 0), after.get(k, 0))
            for k, v in sorted(diff.items()) if v != 0]
    logger.info("After optimization: %s", ', '.join(diff) if diff else "no change")

    return graph
