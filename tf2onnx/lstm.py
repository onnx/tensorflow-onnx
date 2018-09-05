# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.lstm - lstm support
"""

from __future__ import division
from __future__ import print_function

import collections

from tf2onnx.graph_matcher import *


def find_lstm_cells(g, ops):

    xc_pattern = OpTypePattern('Split', inputs=[
        OpTypePattern("BiasAdd", inputs=[
            OpTypePattern("MatMul", inputs=[
                OpTypePattern("ConcatV2|Concat", name="xc"),
                OpTypePattern("*", name="cell_kernel"),
            ]),
            OpTypePattern("*", name="cell_bias"),
        ]),
        OpTypePattern("Const"), # axis for split
    ])

    cell_pattern = \
        OpTypePattern('Mul', name='h', inputs=[
            OpTypePattern("Sigmoid", name="ot", inputs=[xc_pattern]),
            OpTypePattern('Tanh', inputs=[
                OpTypePattern("Add", name="s", inputs=[
                    OpTypePattern("Mul", name="ft", inputs=[
                        OpTypePattern("*", name="prev_state"),
                        OpTypePattern("Sigmoid", inputs=[
                            OpTypePattern("Add", inputs=[
                                "*",
                                xc_pattern,
                            ]),
                        ]),
                    ]),
                    OpTypePattern("Mul", inputs=[
                        OpTypePattern("Sigmoid", name="it", inputs=[xc_pattern]),
                        OpTypePattern("Tanh", name="gt", inputs=[xc_pattern]),
                    ]),
                ]),
            ]),
        ])

    scope_to_cells = collections.defaultdict(list)
    matcher = GraphMatcher(cell_pattern, allow_reorder=True)
    match_results = list(matcher.match_ops(ops))
    # for each found lstm cell, find its outer scope and collect its nodes into a dict.
    for match in match_results:
        # take the cell output and go up 3 levels to find the scope
        h = match.get_op("h")
        parts = h.name.split('/')
        scope = "/".join(parts[0:-3])
        scope_to_cells[scope].extend(match.get_nodes())

    # TODO: rewrite cell to onnx lstm
    #print(scope_to_cells)
    return g.remove_deleted_nodes(ops)
