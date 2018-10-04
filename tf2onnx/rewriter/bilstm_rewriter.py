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

batch_major_ouput_pattern =  \
    OpTypePattern('Pack', name='output_fw_bw_tuple', inputs = [
        OpTypePattern("Transpose", inputs = [
            OpTypePattern("Squeeze", inputs = [
                OpTypePattern("LSTM", name = "lstm_fw")
                # we cannot define inner pattern for LSTM's input, because if some of inputs are const, 
                # that are not in g._nodes, so match pattern search will fail.
            ])
        ]),
        OpTypePattern("Transpose", inputs = [
            OpTypePattern("Squeeze", inputs = [
                OpTypePattern("ReverseV2", inputs = [
                    OpTypePattern("LSTM", name = "lstm_bw"),
                    OpTypePattern("*", name ="reversev2_axis"),
                ]),
            ]),
        ]),
    ])

batch_major_ch_pattern_state_is_not_tuple = \
    OpTypePattern('Pack', name='cell_state_fw_bw_tuple', inputs=[
        OpTypePattern("Squeeze", name="fw_ch_output_squeeze", inputs = [
            OpTypePattern("Concat", inputs = [
                OpTypePattern("LSTM", name = "lstm_fw"),
                OpTypePattern("LSTM", name = "lstm_fw1")
            ]),
        ]),
        OpTypePattern("Squeeze", inputs = [
            OpTypePattern("Concat", inputs = [
                OpTypePattern("LSTM", name = "lstm_bw"),
                OpTypePattern("LSTM", name = "lstm_bw1")
            ]),
        ]),
    ])

batch_major_ch_pattern_state_is_tuple = \
    OpTypePattern('Pack', name='cell_state_fw_bw_tuple', inputs=[
        OpTypePattern("Concat", inputs = [
            OpTypePattern("LSTM", name = "lstm_fw"),
            OpTypePattern("LSTM", name = "lstm_fw1")
        ]),
        OpTypePattern("Concat", inputs = [
            OpTypePattern("LSTM", name = "lstm_bw"),
            OpTypePattern("LSTM", name = "lstm_bw1")
        ]),
    ])

def batch_major_ouput_match_check(g, ops, merged_matches):
    matcher = GraphMatcher(batch_major_ouput_pattern, allow_reorder=False)
    output_match_results = list(matcher.match_ops(ops))
    for match in output_match_results:
        lstm_fw = match.get_op("lstm_fw")
        lstm_bw = match.get_op("lstm_bw")
        transpose_fw = lstm_fw.inputs[0]
        rnn_input_fw = transpose_fw.input[0]

        transpose_bw = lstm_bw.inputs[0]
        reverse_bw = transpose_bw.inputs[0]
        if reverse_bw.type != "ReverseV2":
            continue
        rnn_input_bw = reverse_bw.input[0]

        final_pack = match.get_op("output_fw_bw_tuple")
        left_tran = final_pack.inputs[0]
        right_tran = final_pack.inputs[1]

        if rnn_input_fw == rnn_input_bw and left_tran.type == "Transpose" and right_tran.type == "Transpose":
            fw_bw_pair_name = lstm_fw.name + "," + lstm_bw.name
            if fw_bw_pair_name not in merged_matches:
                merged_matches[fw_bw_pair_name] = [None, None]
                log.debug("batch_major_ouput_match_check: found fw_bw pair " + fw_bw_pair_name)

            assert not merged_matches[fw_bw_pair_name][0]
            to_delete = [transpose_bw, reverse_bw]
            merged_matches[fw_bw_pair_name][0] = MatchedLSTM(transpose_fw.inputs[0], lstm_fw, lstm_bw, final_pack, to_delete, match)
        else:
            continue

def batch_major_ch_match_checks(g, ops, merged_matches):
    batch_major_ch_match_check(g, ops, True , merged_matches)
    batch_major_ch_match_check(g, ops, False, merged_matches)

def batch_major_ch_match_check(g, ops, is_tuple, merged_matches):
    if is_tuple:
        pattern = batch_major_ch_pattern_state_is_tuple
    else:
        pattern = batch_major_ch_pattern_state_is_not_tuple

    # make allow_reorder be False, since we need know which is fw.
    matcher = GraphMatcher(pattern, allow_reorder = False)
    cell_state_match_results = list(matcher.match_ops(ops))
    for match in cell_state_match_results:
        lstm_fw = match.get_op("lstm_fw")
        lstm_bw = match.get_op("lstm_bw")
        if lstm_fw != match.get_op("lstm_fw1") or lstm_bw != match.get_op("lstm_bw1"):
            continue

        transpose_fw = lstm_fw.inputs[0]
        rnn_input_fw = transpose_fw.input[0]

        transpose_bw = lstm_bw.inputs[0]
        reverse_bw = transpose_bw.inputs[0]
        if reverse_bw.type != "ReverseV2":
            continue

        rnn_input_bw = reverse_bw.input[0]
        final_pack = match.get_op("cell_state_fw_bw_tuple")

        if rnn_input_fw == rnn_input_bw:
            fw_bw_pair_name = lstm_fw.name + "," + lstm_bw.name
            if fw_bw_pair_name not in merged_matches:
                merged_matches[fw_bw_pair_name] = [None, None]
                log.debug("batch_major_ch_match_check: add fw_bw pair " + fw_bw_pair_name)

            assert not merged_matches[fw_bw_pair_name][1]
            to_delete = [transpose_bw, reverse_bw]

            matched_lstm = MatchedLSTM(transpose_fw.inputs[0], lstm_fw, lstm_bw, final_pack, to_delete, match)
            matched_lstm.state_is_tuple = is_tuple
            merged_matches[fw_bw_pair_name][1] = matched_lstm
        else:
            continue

# currently we only support inputs having shapes.
# only support batch_major right now.
def process_bilstm_batch_major(g, ops):
    log.info("start rewriting bidirection lstm graph")
    # we use two sub-graph do pattern matching, to make sure we don't make mistakes.
    merged_matches = {}
    batch_major_ouput_match_check(g, ops, merged_matches)
    batch_major_ch_match_checks(g, ops, merged_matches)

    # for each found lstm cell, find its outer scope and collect its nodes into a dict.
    for pair_name in merged_matches:
        log.info("=========================")
        log.info("start handling potential bidirection lstm " + pair_name)
        matches = merged_matches[pair_name]
        lstm_fw = None
        lstm_bw = None
        o_lstm = matches[0]
        ch_lstm = matches[1]
        _lstm = None
        if o_lstm:
            _lstm = o_lstm
        else:
            _lstm = ch_lstm

        lstm_fw = _lstm.lstm_fw
        lstm_bw = _lstm.lstm_bw

        # todo: lstm input might not has a shape
        batch_size = -1
        time_step = -1
        if _lstm.input.shape:
            batch_size = _lstm.input.shape[0]
            time_step = _lstm.input.shape[1]
        else:
            raise ValueError("lstm input does not has shape.")
    
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
        to_append = []
        must_keep_node = []
        if len(lstm_fw.inputs) == len(lstm_bw.inputs):
            if len(lstm_fw.inputs) > 4:
                # todo: non_const seq length
                seq_fw = get_np_val_for_const(g, lstm_fw, 4)
                seq_bw = get_np_val_for_const(g, lstm_bw, 4)
                assert np.array_equal(seq_fw, seq_bw)
                must_keep_node.append(lstm_fw.inputs[0])

                h_node, c_node = process_ch_init_nodes(g, lstm_fw, lstm_bw, to_append)
        else:
            log.error("fw, bw lstm inputs num is not consistent. stop")
            continue

        # create node
        w_name = utils.make_name("W")
        w_node = g.make_const(w_name, W, skip_conversion = True)

        r_name = utils.make_name("R")
        r_node = g.make_const(r_name, R, skip_conversion = True)

        b_name = utils.make_name("B")
        b_node = g.make_const(b_name, B, skip_conversion = True)
        lstm_inputs= [lstm_fw.input[0], w_node.output[0], r_node.output[0], b_node.output[0]]
        if len(lstm_fw.inputs) > 4:
            lstm_inputs.extend([lstm_fw.input[4], h_node.output[0], c_node.output[0]])

        direction = "bidirectional"
        num_directions = 2
        if lstm_fw.get_attr("hidden_size").i == lstm_bw.get_attr("hidden_size").i:
            hidden_size = lstm_fw.get_attr("hidden_size").i
        else:
            log.error("fw and bw has different hidden_size, skip")
            continue

        attr = { "direction": direction, "hidden_size": hidden_size}
        lstm_name = utils.make_name("LSTM")
        lstm_outputs = [lstm_name + ":" + str(i) for i in np.arange(3)]
        bi_lstm_node = Node(helper.make_node("LSTM", lstm_inputs , lstm_outputs, name=lstm_name, **attr), g, skip_conversion = True)
        to_append.append(bi_lstm_node)
        log.info("processing output nodes")
        if o_lstm:
            log.debug("handle o_lstm outputs")
            # we need tranpose LSTM result to original consumer (they assume that)
            # source [seq_length, num_directions, batch_size, hidden_size] 
            # dest [num_directions, batch_size, seq_length, hidden_size]
            new_trans_name = utils.make_name("Transpose")
            attr={ "perm": np.array([1, 2, 0, 3], dtype=np.int64) }
            new_trans = Node(helper.make_node("Transpose", [bi_lstm_node.output[0]], [new_trans_name + ":0"], name=new_trans_name, **attr), g, skip_conversion = True)
            g.set_shape(new_trans.output[0], (num_directions, batch_size, time_step, hidden_size))

            # start to modify graph
            g.replace_all_inputs(all_nodes, o_lstm.output_pack.output[0], new_trans.output[0])
            to_append.append(new_trans)
            for n in o_lstm.nodes_to_delete:
                if n in all_nodes and n not in must_keep_node:
                    all_nodes.remove(n)

        if ch_lstm:
            log.debug("handle ch_lstm outputs")
            # onnx lstm c/h: [num_directions, batch_size, hidden_size]
            if ch_lstm.state_is_tuple:
                # original Pack's output is [num_directions, tuple_size_h_c(e.g. 2), batch_size, hidden_size]
                stack_ch_name = utils.make_name("Pack")
                attr = { "axis": 1 }
                stack_ch = Node(helper.make_node("Pack", [bi_lstm_node.output[2], bi_lstm_node.output[1]], [stack_ch_name + ":0"], name=stack_ch_name, **attr), g, skip_conversion = False)
                g.set_shape(stack_ch.output[0], (num_directions, 2, batch_size, hidden_size))
            else:
                # original Pack's output is [num_directions, batch_size, hidden_size*2]
                stack_ch_name = utils.make_name("Concat")
                attr = { "axis": 2 }
                stack_ch = Node(helper.make_node("Concat", [bi_lstm_node.output[2], bi_lstm_node.output[1]], [stack_ch_name + ":0"], name=stack_ch_name, **attr), g, skip_conversion = True)
                g.set_shape(stack_ch.output[0], (num_directions, batch_size, hidden_size*2))

            to_append.extend([stack_ch])
            log.info(ch_lstm.output_pack.output[0])
            g.replace_all_inputs(all_nodes, ch_lstm.output_pack.output[0], stack_ch.output[0])
            all_nodes.remove(ch_lstm.output_pack)
            for n in ch_lstm.nodes_to_delete:
                if n in all_nodes and n not in must_keep_node:
                    all_nodes.remove(n)
        else:
            print("ch_lstm is None")

        all_nodes.extend(to_append)
        g.set_nodes(all_nodes)

    log.info("done rewriting bidirection lstm graph")
    return g.get_nodes()

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
    init_name = utils.make_name("initial")
    if fw_init_is_const and bw_init_is_const:
        initial_val = np.concatenate((init_fw_val, init_bw_val), axis=0)
        init_node = g.make_const(init_name, initial_val, skip_conversion = True)
    else:
        attr = { "axis" : 0 }
        init_node = Node(helper.make_node("Concat", [fw_init_input_id, bw_init_input_id] , [init_name + ":0"], name=init_name, **attr), g, skip_conversion = True)
        to_append.append(init_node)

    return init_node

def process_ch_init_nodes(g, lstm_fw, lstm_bw, to_append):
    h_node = _process_single_init_node(g, lstm_fw.input[5], lstm_bw.input[5], to_append)
    c_node = _process_single_init_node(g, lstm_fw.input[6], lstm_bw.input[6], to_append)

    return h_node, c_node