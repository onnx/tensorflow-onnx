# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.bigru_rewriter - bigru support.
This rewriter depends on tf2onnx.rewriter.gru_rewriter's results.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.utils import is_tf_reverse_op
from tf2onnx.rewriter.bilstm_rewriter import slice_bilstm_for_original_lstm_consumers,\
     get_reverse_nodes_after_y_output, get_np_val_for_const, _process_single_init_node



logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,unused-argument,missing-docstring

def process_bigru(g, bi_grus):
    for fw, bw in bi_grus:
        input_id = fw[0]
        logger.debug("=========================")
        logger.debug("start handling potential bidirectional gru %s", input_id)

        gru_fw = fw[1]
        gru_bw = bw[1]

        w_fw = get_np_val_for_const(g, gru_fw, 1)
        w_bw = get_np_val_for_const(g, gru_bw, 1)
        r_fw = get_np_val_for_const(g, gru_fw, 2)
        r_bw = get_np_val_for_const(g, gru_bw, 2)
        b_fw = get_np_val_for_const(g, gru_fw, 3)
        b_bw = get_np_val_for_const(g, gru_bw, 3)
        W = np.concatenate((w_fw, w_bw), axis=0)
        R = np.concatenate((r_fw, r_bw), axis=0)
        B = np.concatenate((b_fw, b_bw), axis=0)

        all_nodes = g.get_nodes()
        if len(gru_fw.inputs) == len(gru_bw.inputs):
            if len(gru_fw.inputs) > 4:
                initializer_node = process_init_nodes(g, gru_fw, gru_bw, all_nodes)
        else:
            logger.error("fw, bw gru inputs num is not consistent. stop")
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
        gru_inputs = [gru_fw.input[0], w_node.output[0],
                      r_node.output[0], b_node.output[0]]
        if len(gru_fw.inputs) > 4:
            gru_inputs.extend([gru_fw.input[4], initializer_node.output[0]])

        direction = "bidirectional"
        if gru_fw.get_attr("hidden_size").i == gru_bw.get_attr("hidden_size").i:
            hidden_size = gru_fw.get_attr("hidden_size").i
        else:
            logger.error("fw and bw has different hidden_size, skip")
            continue
        # activation has to be took care
        # attr here is proto
        activations = [act.decode("utf-8")
                       for act in gru_fw.get_attr("activations").strings]
        activations += [act.decode("utf-8")
                        for act in gru_bw.get_attr("activations").strings]
        attr = {"direction": direction, "hidden_size": hidden_size,
                "activations": activations}
        bi_gru_node = g.make_node("GRU", gru_inputs, attr=attr, output_count=2)
        all_nodes.append(bi_gru_node)
        logger.debug("processing output nodes")

        to_remove = [gru_fw.name, gru_fw.input[1], gru_fw.input[2], gru_fw.input[3],
                     gru_bw.name, gru_bw.input[1], gru_bw.input[2], gru_bw.input[3]]
        slice_bilstm_for_original_lstm_consumers(
            g, gru_fw, gru_bw, bi_gru_node, 0, all_nodes, to_remove)
        slice_bilstm_for_original_lstm_consumers(
            g, gru_fw, gru_bw, bi_gru_node, 1, all_nodes, to_remove)

        gru_bw_old_x = gru_bw.input[0]

        for n in to_remove:
            g.remove_node(n)

        old_x_consumers = g.find_output_consumers(gru_bw_old_x)
        # the transpose/reverse here must be followed by GRU if it is still useful.
        # this is guaranteed by dynamic_rnn logic.
        old_x_has_gru_as_consumer = [
            n for n in old_x_consumers if n.type == "GRU"]
        if not old_x_has_gru_as_consumer:
            logger.debug("plan to remove useless reverse op in bw")
            reverse_node = g.get_node_by_output(gru_bw_old_x)

            if reverse_node.type == "Transpose":
                reverse_node = reverse_node.inputs[0]

            g.replace_all_inputs(
                g.get_nodes(), reverse_node.output[0], reverse_node.input[0])
            g.remove_node(reverse_node.name)
        else:
            raise ValueError(
                "Reverse is still used by GRU as input, cannot remove")

    return g.get_nodes()


def process_init_nodes(g, gru_fw, gru_bw, to_append):
    initializer_node = _process_single_init_node(
        g, gru_fw.input[5], gru_bw.input[5], to_append)

    return initializer_node


def rewrite_bidirectional_grus(g, ops):
    """
        return: list of tuple, format of tuple is
        ((fw input_id, fw onnx gru node), (bw input_id, bw onnx gru node)), and fw input_id equals to bw input_id
    """
    fw_gru = {}
    bw_gru = {}
    for n in g.get_nodes():
        if n.type != "GRU":
            continue
        input_id = n.input[0]
        temp = n.inputs[0]
        is_backward_gru = False
        if temp.type == "Transpose":
            input_id = temp.input[0]
            temp = temp.inputs[0]

        if is_tf_reverse_op(temp):
            input_id = temp.input[0]
            is_backward_gru = True

        if is_backward_gru:
            # if output 0 is consumed, and there is no reverse after the gru output.
            # it's not reversed gru
            if g.find_output_consumers(n.output[0]) and not get_reverse_nodes_after_y_output(g, n):
                continue
            logger.debug("find bw gru %s", input_id)
            bw_gru[input_id] = [input_id, n]
        else:
            logger.debug("find fw gru %s", input_id)
            fw_gru[input_id] = [input_id, n]

    # when fw_gru has same input as bw_gru, then it may be a bi gru
    bigru_input = list(set(fw_gru.keys()).intersection(bw_gru.keys()))
    bi_grus = [(fw_gru[input_id], bw_gru[input_id])
               for input_id in bigru_input]

    return process_bigru(g, bi_grus)
