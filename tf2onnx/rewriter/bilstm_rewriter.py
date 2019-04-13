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
from tf2onnx.rewriter.rnn_utils import is_reverse_op


logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,unused-argument,missing-docstring

def process_bilstm(g, bi_lstms):
    for fw, bw in bi_lstms:
        input_id = fw[0]
        logger.debug("=========================")
        logger.debug("start handling potential bidirectional lstm %s", input_id)

        lstm_fw = fw[1]
        lstm_bw = bw[1]

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
        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 0, all_nodes, to_remove)
        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 1, all_nodes, to_remove)
        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 2, all_nodes, to_remove)

        lstm_bw_old_x = lstm_bw.input[0]
        for n in to_remove:
            g.remove_node(n)

        old_x_consumers = g.find_output_consumers(lstm_bw_old_x)
        # the transpose/reverse here must be followed by LSTM if it is still useful.
        # this is guaranteed by dynamic_rnn logic.
        old_x_has_lstm_as_consumer = [n for n in old_x_consumers if n.type == "LSTM"]
        if not old_x_has_lstm_as_consumer:
            logger.debug("plan to remove useless reverse op in bw")
            reverse_node = g.get_node_by_output(lstm_bw_old_x)

            if reverse_node.type == "Transpose":
                reverse_node = reverse_node.inputs[0]

            g.replace_all_inputs(g.get_nodes(), reverse_node.output[0], reverse_node.input[0])
            g.remove_node(reverse_node.name)
        else:
            raise ValueError("Reverse is still used by LSTM as input, cannot remove")

    return g.get_nodes()


def slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm, lstm_output_index, all_nodes, to_remove):
    fw_consumers = g.find_output_consumers(lstm_fw.output[lstm_output_index])
    bw_consumers = g.find_output_consumers(lstm_bw.output[lstm_output_index])
    if not fw_consumers and not bw_consumers:
        return

    if lstm_output_index == 0:
        axis = 1
        # remove reverse op for lstm_bw
        reverse_nodes = get_reverse_nodes_after_y_output(g, lstm_bw)
        if not reverse_nodes:
            raise ValueError("should not happen y_output is not followed with reverse node")

        for r_op in reverse_nodes:
            logger.debug("remove reverse op called %s", r_op.name)
            g.replace_all_inputs(all_nodes, r_op.output[0], r_op.input[0])
            to_remove.append(r_op.name)
    elif lstm_output_index in [1, 2]:
        axis = 0
    else:
        raise ValueError("LSTM only should has 3 outputs.")

    if fw_consumers:
        attr = {"axes": [axis], "starts": [0], "ends": [1]}
        slice_node_fw = g.make_node("Slice", [bi_lstm.output[lstm_output_index]], attr=attr)
        all_nodes.append(slice_node_fw)
        g.replace_all_inputs(fw_consumers, lstm_fw.output[lstm_output_index], slice_node_fw.output[0])

    if bw_consumers:
        attr = {"axes": [axis], "starts": [1], "ends": [2]}
        slice_node_bw = g.make_node("Slice", [bi_lstm.output[lstm_output_index]], attr=attr)
        all_nodes.append(slice_node_bw)
        g.replace_all_inputs(bw_consumers, lstm_bw.output[lstm_output_index], slice_node_bw.output[0])


def check_const(g, input_id):
    node = g.get_node_by_output(input_id)
    if node and node.is_const():
        return (True, node.get_tensor_value(as_list=False))
    return (None, None)


def get_np_val_for_const(g, node, input_index):
    return node.inputs[input_index].get_tensor_value(as_list=False)


def _process_single_init_node(g, fw_init_input_id, bw_init_input_id, to_append):
    fw_init_is_const, init_fw_val = check_const(g, fw_init_input_id)
    bw_init_is_const, init_bw_val = check_const(g, bw_init_input_id)
    if fw_init_is_const and bw_init_is_const:
        initial_val = np.concatenate((init_fw_val, init_bw_val), axis=0)
        init_name = utils.make_name("initial")
        init_node = g.make_const(init_name, initial_val, skip_conversion=True)
    else:
        init_node = g.make_node("Concat", [fw_init_input_id, bw_init_input_id], attr={"axis": 0})

    to_append.append(init_node)
    return init_node


def process_ch_init_nodes(g, lstm_fw, lstm_bw, to_append):
    h_node = _process_single_init_node(g, lstm_fw.input[5], lstm_bw.input[5], to_append)
    c_node = _process_single_init_node(g, lstm_fw.input[6], lstm_bw.input[6], to_append)

    return h_node, c_node


def rewrite_bidirectional_lstms(g, ops):
    fw_lstm = {}
    bw_lstm = {}
    for n in g.get_nodes():
        if n.type != "LSTM":
            continue
        input_id = n.input[0]
        temp = n.inputs[0]
        is_backward_lstm = False
        if temp.type == "Transpose":
            input_id = temp.input[0]
            temp = temp.inputs[0]

        if is_reverse_op(temp):
            input_id = temp.input[0]
            is_backward_lstm = True

        if is_backward_lstm:
            # if output 0 is consumed, and there is no reverse after the lstm output.
            # it's not reversed lstm
            if g.find_output_consumers(n.output[0]) and not get_reverse_nodes_after_y_output(g, n):
                continue

            logger.debug("find bw lstm %s", input_id)
            bw_lstm[input_id] = [input_id, n]
        else:
            logger.debug("find fw lstm %s", input_id)
            fw_lstm[input_id] = [input_id, n]

    bilstm_input = list(set(fw_lstm.keys()).intersection(bw_lstm.keys()))
    bi_lstms = [(fw_lstm[input_id], bw_lstm[input_id]) for input_id in bilstm_input]

    return process_bilstm(g, bi_lstms)


def get_reverse_nodes_after_y_output(g, lstm_bw):
    bw_consumers = g.find_output_consumers(lstm_bw.output[0])

    # todo: figure out a better way to remove reverse op
    squeeze_nodes = [c for c in bw_consumers if c.type == "Squeeze"]
    s_cnt = len(squeeze_nodes)
    if s_cnt == 1:
        s = squeeze_nodes[0]
        trans_nodes = g.find_output_consumers(s.output[0])
        if len(trans_nodes) == 1:
            if trans_nodes[0].type == "Transpose":
                reverse_nodes = g.find_output_consumers(trans_nodes[0].output[0])
            elif is_reverse_op(trans_nodes[0]):
                reverse_nodes = trans_nodes
            else:
                logger.debug("not found reverse op, unexpected")
                return None

            are_all_reverse = all([is_reverse_op(r_op) for r_op in reverse_nodes])
            if are_all_reverse:
                return reverse_nodes

            logger.debug("bw y output is used followed by reverse node")
            return None

        logger.debug("unexpected number of transpose after LSTM 1st output:%s", s_cnt)
        return None

    logger.debug("unexpected number of squeeze following LSTM 1st output:%s", s_cnt)
    return None
