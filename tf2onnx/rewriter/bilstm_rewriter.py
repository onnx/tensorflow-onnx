# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.bilstm_rewriter - bilstm support.
This rewriter depends on tf2onnx.rewriter.lstm_rewriter's results.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.rewriter.rnn_utils import find_bidirectional_rnns, ONNX_RNN_TYPE, \
    get_np_val_for_const, process_single_init_node, slice_birnn_for_original_rnn_consumers

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,unused-argument,missing-docstring

def process_bilstm(g, bi_lstms):
    for lstm_fw, lstm_bw in bi_lstms:
        logger.debug("=========================")
        logger.debug("start handling potential bidirectional lstm: %s, %s", lstm_fw.name, lstm_bw.name)

        w_fw = get_np_val_for_const(g, lstm_fw, 1)
        w_bw = get_np_val_for_const(g, lstm_bw, 1)
        r_fw = get_np_val_for_const(g, lstm_fw, 2)
        r_bw = get_np_val_for_const(g, lstm_bw, 2)
        b_fw = get_np_val_for_const(g, lstm_fw, 3)
        b_bw = get_np_val_for_const(g, lstm_bw, 3)
        W = np.concatenate((w_fw, w_bw), axis=0)
        R = np.concatenate((r_fw, r_bw), axis=0)
        B = np.concatenate((b_fw, b_bw), axis=0)

        all_nodes = g.get_nodes()
        if len(lstm_fw.inputs) == len(lstm_bw.inputs):
            if len(lstm_fw.inputs) > 4:
                h_node, c_node = process_ch_init_nodes(g, lstm_fw, lstm_bw, all_nodes)
        else:
            logger.error("fw, bw lstm inputs num is not consistent. stop")
            continue

        # create node
        w_name = utils.make_name("W")
        w_node = g.make_const(w_name, W, skip_conversion=True)
        all_nodes.append(w_node)

        r_name = utils.make_name("R")
        r_node = g.make_const(r_name, R, skip_conversion=True)
        all_nodes.append(r_node)

        b_name = utils.make_name("B")
        b_node = g.make_const(b_name, B, skip_conversion=True)
        all_nodes.append(b_node)
        lstm_inputs = [lstm_fw.input[0], w_node.output[0], r_node.output[0], b_node.output[0]]
        if len(lstm_fw.inputs) > 4:
            lstm_inputs.extend([lstm_fw.input[4], h_node.output[0], c_node.output[0]])

        direction = "bidirectional"
        if lstm_fw.get_attr("hidden_size").i == lstm_bw.get_attr("hidden_size").i:
            hidden_size = lstm_fw.get_attr("hidden_size").i
        else:
            logger.error("fw and bw has different hidden_size, skip")
            continue

        attr = {"direction": direction, "hidden_size": hidden_size}
        bi_lstm_node = g.make_node("LSTM", lstm_inputs, attr=attr, output_count=3)
        all_nodes.append(bi_lstm_node)
        logger.debug("processing output nodes")

        to_remove = [lstm_fw.name, lstm_fw.input[1], lstm_fw.input[2], lstm_fw.input[3],
                     lstm_bw.name, lstm_bw.input[1], lstm_bw.input[2], lstm_bw.input[3]]
        slice_birnn_for_original_rnn_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 0, all_nodes, to_remove)
        slice_birnn_for_original_rnn_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 1, all_nodes, to_remove)
        slice_birnn_for_original_rnn_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 2, all_nodes, to_remove)

        for n in to_remove:
            g.remove_node(n)

    return g.get_nodes()


def process_ch_init_nodes(g, lstm_fw, lstm_bw, to_append):
    h_node = process_single_init_node(g, lstm_fw.input[5], lstm_bw.input[5], to_append)
    c_node = process_single_init_node(g, lstm_fw.input[6], lstm_bw.input[6], to_append)

    return h_node, c_node


def rewrite_bidirectional_lstms(g, ops):
    bi_lstms = find_bidirectional_rnns(g, ops, ONNX_RNN_TYPE.LSTM)

    return process_bilstm(g, bi_lstms)
