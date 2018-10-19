# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.gru_rewriter - gru support
"""

from __future__ import division
from __future__ import print_function

from tf2onnx.rewriter.unit_rewriter_base import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.gru_rewriter")


class GRUUnitRewriter(UnitRewriterBase):
    def __init__(self, g):
        super(GRUUnitRewriter, self).__init__(g)

    def run(self):
        return super(GRUUnitRewriter, self).run(RNNUnitType.GRUCell)

    def _ht_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        raise ValueError("not implemented")

    def _output_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        raise ValueError("not implemented")

    def process_weights_and_bias(self, rnn_weights):
        raise ValueError("not implemented")

    def get_rnn_scope_name(self, match):
        raise ValueError("not implemented")

    def get_weight_and_bias(self, match):
        raise ValueError("not implemented")

    def process_input_x(self, rnn_props, rnn_scope_name):
        raise ValueError("not implemented")

    def process_var_init_nodes(self, rnn_props):
        raise ValueError("not implemented")

    def process_seq_length(self, rnn_props, seq_len_input_node):
        raise ValueError("not implemented")

    def create_rnn_node(self, rnn_props):
        raise ValueError("not implemented")
