# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.bilstm_rewriter - bilstm support.
This rewriter depends on tf2onnx.rewriter.lstm_rewriter's results.
"""

from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tf2onnx

from onnx import helper, defs, numpy_helper, checker, onnx_pb
from onnx import AttributeProto, TensorProto, GraphProto
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import *
from tf2onnx.rewriter.rnn_utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.bilstm_rewriter")


def process_bilstm(g, bi_lstms):
    for fw, bw in bi_lstms:
        input_id = fw[0]
        log.info("=========================")
        log.info("start handling potential bidirectional lstm " + input_id)

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
            log.error("fw, bw lstm inputs num is not consistent. stop")
            continue

        # create node
        w_name = utils.make_name("W")
        w_node = g.make_const(w_name, W, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = g.make_const(r_name, R, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = g.make_const(b_name, B, skip_conversion=True)
        lstm_inputs= [lstm_fw.input[0], w_node.output[0], r_node.output[0], b_node.output[0]]
        if len(lstm_fw.inputs) > 4:
            lstm_inputs.extend([lstm_fw.input[4], h_node.output[0], c_node.output[0]])

        direction = "bidirectional"
        if lstm_fw.get_attr("hidden_size").i == lstm_bw.get_attr("hidden_size").i:
            hidden_size = lstm_fw.get_attr("hidden_size").i
        else:
            log.error("fw and bw has different hidden_size, skip")
            continue

        attr = {"direction": direction, "hidden_size": hidden_size}
        bi_lstm_node = make_onnx_node(g, "LSTM", lstm_inputs, attr=attr, output_count=3)
        all_nodes.append(bi_lstm_node)
        log.info("processing output nodes")

        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 0, all_nodes)
        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 1, all_nodes)
        slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm_node, 2, all_nodes)

        reverse_ops = [op for op in all_nodes if is_reverse_op(op)]
        for r_op in reverse_ops:
            g.replace_all_inputs(all_nodes, r_op.output[0], r_op.input[0])

        to_remove = [lstm_fw.name, lstm_fw.input[1], lstm_fw.input[2], lstm_fw.input[3], 
            lstm_fw.input[5], lstm_fw.input[6], lstm_bw.name, lstm_bw.input[1],
            lstm_bw.input[2], lstm_bw.input[3], lstm_bw.input[5], lstm_bw.input[6]]
        to_remove.extend([r_op.name for r_op in reverse_ops])
        new_nodes = []
        for n in all_nodes:
            if n.name not in to_remove:
                new_nodes.append(n)
        g.set_nodes(new_nodes)
    return g.get_nodes()


def slice_bilstm_for_original_lstm_consumers(g, lstm_fw, lstm_bw, bi_lstm, lstm_output_index, all_nodes):
    fw_consumers = g.find_output_consumers(lstm_fw.output[lstm_output_index])
    bw_consumers = g.find_output_consumers(lstm_bw.output[lstm_output_index])

    if lstm_output_index == 0:
        axis = 1
    elif lstm_output_index == 1 or lstm_output_index == 2:
        axis = 0
    else:
        raise ValueError("LSTM only should has 3 outputs.")

    if fw_consumers:
        attr = {"axes": [axis], "starts": [0], "ends": [1]}
        slice_node_fw = make_onnx_node(g, "Slice", [bi_lstm.output[lstm_output_index]], attr)
        all_nodes.append(slice_node_fw)
        g.replace_all_inputs(fw_consumers, lstm_fw.output[lstm_output_index], slice_node_fw.output[0])

    if bw_consumers:
        attr = {"axes": [axis], "starts": [1], "ends": [2]}
        slice_node_bw = make_onnx_node(g, "Slice", [bi_lstm.output[lstm_output_index]], attr)
        all_nodes.append(slice_node_bw)
        g.replace_all_inputs(bw_consumers, lstm_bw.output[lstm_output_index], slice_node_bw.output[0])


def check_const(g, input_id):
    node = g.get_node_by_name(input_id)
    if node and node.is_const():
        return (True, node.get_tensor_value())
    elif g.is_initializer(input_id):
        tensor = g.get_initializer(input_id)
        return (True, numpy_helper.to_array(tensor))

    return (None, None)


def get_np_val_for_const(g, node, input_index):
    input_name = node.input[input_index]
    tensor = g.get_initializer(input_name)
    return numpy_helper.to_array(tensor)


def _process_single_init_node(g, fw_init_input_id, bw_init_input_id, to_append):
    fw_init_is_const, init_fw_val = check_const(g, fw_init_input_id)
    bw_init_is_const, init_bw_val = check_const(g, bw_init_input_id)
    if fw_init_is_const and bw_init_is_const:
        initial_val = np.concatenate((init_fw_val, init_bw_val), axis=0)
        init_name = utils.make_name("initial")
        init_node = g.make_const(init_name, initial_val, skip_conversion = True)
    else:
        attr = {"axis" : 0}
        init_node = make_onnx_node(g, "Concat", [fw_init_input_id, bw_init_input_id], attr)
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
            log.info("find bw lstm" + input_id)
            bw_lstm[input_id] = [input_id, n]
        else:
            log.info("find fw lstm" + input_id)
            fw_lstm[input_id] = [input_id, n]

    bilstm_input = list(set(fw_lstm.keys()).intersection(bw_lstm.keys()))
    bi_lstms = [(fw_lstm[input_id], bw_lstm[input_id]) for input_id in bilstm_input]

    return process_bilstm(g, bi_lstms)
