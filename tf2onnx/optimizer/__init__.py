# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
"""tf2onnx.optimizer module"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import traceback
from collections import OrderedDict

from tf2onnx.optimizer.const_fold_optimizer import ConstFoldOptimizer
from tf2onnx.optimizer.identity_optimizer import IdentityOptimizer
from tf2onnx.optimizer.merge_duplicated_nodes_optimizer import MergeDuplicatedNodesOptimizer
from tf2onnx.optimizer.transpose_optimizer import TransposeOptimizer

# pylint: disable=missing-docstring, broad-except

# optimizer sequence need to be considered carefully
_optimizers = OrderedDict([
    ("transpose_opt", TransposeOptimizer),
    ("fold_const", ConstFoldOptimizer),
    # merge_duplicated_nodes should be used after transpose_opt
    # for transpose_opt may have some trans nodes that can be merge
    ("merge_duplicated_nodes", MergeDuplicatedNodesOptimizer),
    ("identity_opt", IdentityOptimizer),
])


def optimize_graph(graph):
    try:
        opts = _get_optimizers()
        for opt in opts.values():
            graph = opt().optimize(graph)

        graph.update_proto()
        return graph
    except Exception:
        # degradation to non-optimized model proto
        type_, value_, traceback_ = sys.exc_info()
        ex_ext = traceback.format_exception(type_, value_, traceback_)
        print("NON-CRITICAL error in optimizer: ", ex_ext)
        return None


def _get_optimizers():
    return _optimizers
