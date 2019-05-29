# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn_utils - rnn support
"""

from __future__ import unicode_literals
from collections import defaultdict
from enum import Enum

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher # pylint: disable=unused-import


# pylint: disable=invalid-name,unused-argument,missing-docstring



logger = logging.getLogger(__name__)


class REWRITER_RESULT(Enum):
    SKIP = 1
    OK = 2
    FAIL = 3


# TensorFlow LSTMCell/BasicLSTMCell computation graph matching
xc_pattern = OpTypePattern('Split', inputs=[
    OpTypePattern("Const"), # axis for split
    OpTypePattern("BiasAdd", name="bias_add", inputs=[
        OpTypePattern("MatMul", inputs=[
            OpTypePattern("ConcatV2|Concat", name="xh"),
            OpTypePattern("Enter", inputs=[
                OpTypePattern("*", name="cell_kernel"),
            ]),
        ]),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_bias"),
        ]),
    ]),
])


lstmcell_pattern = \
    OpTypePattern('Mul', name='ht', inputs=[
        OpTypePattern("Sigmoid", name="ot", inputs=[xc_pattern]),
        OpTypePattern('Tanh', inputs=[
            OpTypePattern("Add", name="ct", inputs=[
                OpTypePattern("Mul", name="ct_identity_consumer", inputs=[
                    OpTypePattern("Sigmoid", name="ft", inputs=[
                        OpTypePattern("Add", inputs=[
                            xc_pattern,
                            OpTypePattern("*", name="ft_bias"),
                        ]),
                    ]),
                    OpTypePattern("*"),
                ]),
                OpTypePattern("Mul", inputs=[
                    OpTypePattern("Sigmoid", name="it", inputs=[xc_pattern]),
                    OpTypePattern("Tanh", name="gt", inputs=[xc_pattern]),
                ]),
            ]),
        ]),
    ])

# input sequence: top to down, left to right
# split into update gate and reset gate
gru_split_pattern = \
    OpTypePattern("Split", inputs=[
        OpTypePattern("Const"),  # split dim, a constant
        OpTypePattern("Sigmoid", inputs=[
            OpTypePattern("BiasAdd", inputs=[
                OpTypePattern("Enter", inputs=[
                    OpTypePattern("*", name="gate_bias")
                ]),
                OpTypePattern("MatMul", name="update_reset_gate", inputs=[
                    OpTypePattern("Enter", inputs=[
                        OpTypePattern("*", name="gate_kernel")
                    ]),
                    OpTypePattern("ConcatV2|Concat", name="cell_inputs")
                ])
            ])
        ])
    ])


grucell_pattern = \
    OpTypePattern("Add", name="cell_output", inputs=[
        OpTypePattern("Mul", inputs=[
            gru_split_pattern,
            OpTypePattern("Identity")
        ]),
        OpTypePattern("Mul", inputs=[
            OpTypePattern("Sub", inputs=[
                OpTypePattern("Const"),  # 1-u
                gru_split_pattern
            ]),
            OpTypePattern("*", name="optional_activation", inputs=[
                OpTypePattern("BiasAdd", inputs=[
                    OpTypePattern("Enter", inputs=[
                        OpTypePattern("*", name="hidden_bias")
                    ]),
                    OpTypePattern("MatMul", inputs=[
                        OpTypePattern("Enter", inputs=[
                            OpTypePattern("*", name="hidden_kernel")
                        ]),
                        OpTypePattern("ConcatV2|Concat")
                    ])
                ])
            ])
        ])
    ])


cudnn_compatible_grucell_pattern = \
    OpTypePattern("Add", name="cell_output", inputs=[
        OpTypePattern("Mul", inputs=[
            OpTypePattern("Sub", inputs=[
                OpTypePattern("Const"),  # 1-u
                gru_split_pattern
            ]),
            OpTypePattern("*", name="optional_activation", inputs=[
                OpTypePattern("Add", inputs=[
                    OpTypePattern("Mul", inputs=[
                        gru_split_pattern,
                        OpTypePattern("BiasAdd", inputs=[
                            OpTypePattern("Enter", inputs=[
                                OpTypePattern("*", name="hidden_state_bias")
                            ]),
                            OpTypePattern("MatMul", inputs=[
                                OpTypePattern("Enter", inputs=[
                                    OpTypePattern("*", name="hidden_state_kernel"),
                                ]),
                                OpTypePattern("Identity")
                            ])
                        ])
                    ]),
                    OpTypePattern("BiasAdd", inputs=[
                        OpTypePattern("Enter", inputs=[
                            OpTypePattern("*", name="hidden_input_bias")
                        ]),
                        OpTypePattern("MatMul", inputs=[
                            OpTypePattern("Enter", inputs=[
                                OpTypePattern("*", name="hidden_input_kernel"),
                            ]),
                            OpTypePattern("*")
                        ])
                    ])
                ])
            ])
        ]),
        OpTypePattern("Mul", inputs=[
            gru_split_pattern,
            OpTypePattern("Identity")
        ])
    ])


grublockcell_pattern = OpTypePattern("GRUBlockCell", name="gru_block_cell", inputs=[
    OpTypePattern("*"),
    OpTypePattern("*"),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="gate_kernel")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="hidden_kernel")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="gate_bias")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="hidden_bias")
    ])
])


lstmblockcell_pattern = \
    OpTypePattern("LSTMBlockCell", name="lstm_block_cell", inputs=[
        OpTypePattern("*"),
        OpTypePattern("*"),
        OpTypePattern("*"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_kernel")
        ]),
        OpTypePattern("*", name="Pi"),
        OpTypePattern("*", name="Pf"),
        OpTypePattern("*", name="Po"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_bias")
        ])
    ])



seq_len_pattern = OpTypePattern("Select", inputs=[
    OpTypePattern("GreaterEqual", inputs=[
        OpTypePattern("*"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="seq_len_node")
        ])
    ]),
    OpTypePattern("*"),
    OpTypePattern("*")
])


class RNNUnitType(Enum):
    LSTMCell = 0  # TF LSTMCell and BasicLSTMCell share the same pattern
    LSTMBlockCell = 1
    GRUCell = 2
    GRUBlockCell = 3
    CudnnCompatibleGRUCell = 4


rnn_cell_patterns = {
    RNNUnitType.LSTMCell: lstmcell_pattern,
    RNNUnitType.LSTMBlockCell: lstmblockcell_pattern,
    RNNUnitType.GRUCell: grucell_pattern,
    RNNUnitType.GRUBlockCell: grublockcell_pattern,
    RNNUnitType.CudnnCompatibleGRUCell: cudnn_compatible_grucell_pattern
}


def get_pattern(cell_type_name):
    return rnn_cell_patterns[cell_type_name]


def get_rnn_scope_name(while_scope_name):
    parts = while_scope_name.split('/')
    rnn_scope = '/'.join(parts[0:-2]) + "/"
    return rnn_scope


def parse_rnn_loop(graph, loop_properties, rnn_scope, while_context_scope):
    """check if the while loop is generated by dynamic_rnn or bidirectional_rnn

    Args:
        loop_properties: LoopProperties
        rnn_scope: rnn scope name
        while_context_scope: while loop scope name
    """
    # check a while loop is generated by dynamic_rnn or bidirectional_rnn by
    #
    # 1. some patterns in _time_step in dynamic_rnn: tensor array read, tensor array write
    # 2. some patterns in control_flow_ops.while_loop in dynamic_rnn:
    #      cond: time < loop_bound
    #      loop_vars: (time, output_ta, state)
    #      time has name called "time"
    #      iteration_cnt is added by control flow.

    # be noted:
    # 1. iteration counter does not exist in tf1.4 or earlier versions
    # 2. if dynamic_rnn's first input is not consumed, output ta does not exist.
    time_name = rnn_scope + "time"
    ta_array_name_prefix = rnn_scope + "dynamic_rnn/output_"
    iteration_counter_name = while_context_scope + "iteration_counter"

    found_time = False
    is_rnn_out_ta = None
    time_var = None
    iteration_var = None
    for val in loop_properties.all_variables.values():
        enter_input_node = graph.get_node_by_output(val.enter_input_id)
        if val.is_tensor_array:
            ta_name = enter_input_node.get_attr("tensor_array_name").s.decode("utf-8")
            if not ta_name.startswith(ta_array_name_prefix):
                is_rnn_out_ta = False
        elif enter_input_node.name == time_name:
            found_time = True
            time_var = val
        elif enter_input_node.name == iteration_counter_name:
            iteration_var = val

    if not found_time or is_rnn_out_ta is False:
        logger.debug("this should not be a dynamic_rnn loop, found_time: %s, is_rnn_out_ta: %s",
                     found_time, is_rnn_out_ta)
        return None

    if not loop_properties.tensor_array_inputs:
        logger.debug("this should not be a dynamic_rnn loop, no ta input is found")
        return None

    return time_var, iteration_var


def get_weights_from_const_node(g, node):
    temp = node
    val = None
    # this would help ignore Identity in non-const_folded graph.
    while temp.type == 'Identity':
        temp = temp.inputs[0]

    if temp and temp.type == 'Const':
        val = temp.get_tensor_value(as_list=False)
        dtype = utils.map_onnx_to_numpy_type(g.get_dtype(temp.output[0]))
        val = val.astype(dtype)
        logger.debug("found weights %s", temp.name)
    else:
        logger.debug("weight node seems not to be Const, skip, node name is %s", temp.name)
        return None

    return val


######################################################
####      Utilities for bidirectional rnn      #######
######################################################
class ONNX_RNN_TYPE(Enum):
    GRU = 0
    LSTM = 1


onnx_rnn_type_mapping = {
    ONNX_RNN_TYPE.GRU: "GRU",
    ONNX_RNN_TYPE.LSTM: "LSTM"
}


def find_bidirectional_rnns(g, ops, rnn_type):
    """
    find possible bidirectional rnns, return: list of tuple,
    format of tuple is (fw onnx rnn node, bw onnx rnn node).
    """
    fw_rnns = defaultdict(list)
    bw_rnns = defaultdict(list)
    for n in g.get_nodes():
        if n.type != onnx_rnn_type_mapping[rnn_type]:
            continue

        input_id = n.input[0]
        temp = n.inputs[0]
        is_bw_from_input = False
        if temp.type == "Transpose":
            input_id = temp.input[0]
            temp = temp.inputs[0]

        if utils.is_reverse_op(temp):
            input_id = temp.input[0]
            is_bw_from_input = True

        if is_bw_from_input:
            logger.debug("find bw rnn %s", input_id)
            bw_rnns[input_id].append(n)
        else:
            logger.debug("find fw rnn %s", input_id)
            fw_rnns[input_id].append(n)

    # when fw_rnn has same input as bw_rnn, then it may be a bi rnn
    birnn_input = list(set(fw_rnns.keys()).intersection(bw_rnns.keys()))
    bi_rnns = []
    for inp in birnn_input:
        fw_rnn = fw_rnns[inp]
        bw_rnn = bw_rnns[inp]
        # when bi-rnn 1st output are unused, only single birnn is support
        if len(fw_rnn) == 1 and len(bw_rnn) == 1 and \
                not g.find_output_consumers(fw_rnn[0].output[0]) and \
                not g.find_output_consumers(bw_rnn[0].output[0]):
            bi_rnns.append((fw_rnn[0], bw_rnn[0]))
            continue

        for fw_n in fw_rnn:
            for bw_n in bw_rnn:
                if belong_to_birnn(g, fw_n, bw_n):
                    logger.debug("found birnn comprising %s and %s", fw_n.name, bw_n.name)
                    bi_rnns.append((fw_n, bw_n))
    return bi_rnns


def belong_to_birnn(g, fw_rnn, bw_rnn):
    """check fw_rnn, bw_rnn are part of the same birnn"""
    if not g.find_output_consumers(fw_rnn.output[0]) or \
            not g.find_output_consumers(bw_rnn.output[0]):
        logger.warning("cannot support birnn with unused 1st output, please check them by yourself.")
        return False

    is_bw_for_fw, fw_output_id = check_birnn_output_after_y(g, fw_rnn)
    if is_bw_for_fw:
        logger.warning("forward rnn %s is followed by Reverse op", fw_rnn.name)
        return False
    is_bw_for_bw, bw_output_id = check_birnn_output_after_y(g, bw_rnn)
    if not is_bw_for_bw and bw_output_id:
        logger.warning("backward rnn %s isn't followed by Reverse op", bw_rnn.name)
        return False

    # pair of bi-rnn must share the same concat or pack as output consumer
    common_consumer = g.find_common_consumers(fw_output_id, bw_output_id)
    if len(common_consumer) == 1 and \
            (utils.is_concat_op(common_consumer[0]) or common_consumer[0].type == "Pack"):
        return True
    return False


def check_birnn_output_after_y(g, rnn):
    """
    get the output following the 1st output of bi-rnn op before Concat or Pack op.
    and check if rnn is bw, that is, followed by Reverse
    """
    is_bw = False
    real_output = rnn
    bw_consumers = g.find_output_consumers(rnn.output[0])
    if not bw_consumers:
        logger.debug("no one consume 1st output of %s", rnn.name)
        return is_bw, real_output

    squeeze_nodes = [c for c in bw_consumers if c.type == "Squeeze"]
    s_cnt = len(squeeze_nodes)
    if s_cnt == 1:
        s = squeeze_nodes[0]
        trans_nodes = g.find_output_consumers(s.output[0])
        if len(trans_nodes) == 1 and trans_nodes[0].type == "Transpose":
            real_output = trans_nodes[0].output[0]
        else:
            real_output = s.output[0]
    else:
        logger.debug("unexpected number of squeeze following RNN 1st output:%s", s_cnt)

    reverse_nodes = g.find_output_consumers(real_output)
    if len(reverse_nodes) == 1 and utils.is_reverse_op(reverse_nodes[0]):
        is_bw = True
        real_output = reverse_nodes[0].output[0]

    return is_bw, real_output


def get_np_val_for_const(g, node, input_index):
    return node.inputs[input_index].get_tensor_value(as_list=False)


def check_const(g, input_id):
    node = g.get_node_by_output(input_id)
    if node and node.is_const():
        return (True, node.get_tensor_value(as_list=False))
    return (None, None)


def process_single_init_node(g, fw_init_input_id, bw_init_input_id, to_append):
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


def slice_birnn_for_original_rnn_consumers(g, rnn_fw, rnn_bw, bi_rnn, rnn_output_index, all_nodes, to_remove):
    fw_consumers = g.find_output_consumers(rnn_fw.output[rnn_output_index])
    bw_consumers = g.find_output_consumers(rnn_bw.output[rnn_output_index])
    if not fw_consumers and not bw_consumers:
        return

    if rnn_output_index == 0:
        axis = 1
        # remove reverse op for rnn_bw
        _, real_output = check_birnn_output_after_y(g, rnn_bw)
        reverse_node = g.get_node_by_output(real_output)
        if not reverse_node or not utils.is_reverse_op(reverse_node):
            raise ValueError("should not happen y_output is not followed with reverse node")

        logger.debug("remove reverse op called %s", reverse_node.name)
        g.replace_all_inputs(all_nodes, reverse_node.output[0], reverse_node.input[0])
        to_remove.append(reverse_node.name)
    elif rnn_output_index in [1, 2]:
        axis = 0
    else:
        raise ValueError("rnn only should has 3 outputs.")

    if fw_consumers:
        attr = {"axes": [axis], "starts": [0], "ends": [1]}
        inputs_map = {"data": bi_rnn.output[rnn_output_index], **attr}
        slice_node_fw = GraphBuilder(g).make_slice(inputs_map)
        all_nodes.append(g.get_node_by_output(slice_node_fw))
        g.replace_all_inputs(fw_consumers, rnn_fw.output[rnn_output_index], slice_node_fw)

    if bw_consumers:
        attr = {"axes": [axis], "starts": [1], "ends": [2]}
        inputs_map = {"data": bi_rnn.output[rnn_output_index], **attr}
        slice_node_bw = GraphBuilder(g).make_slice(inputs_map)
        all_nodes.append(g.get_node_by_output(slice_node_bw))
        g.replace_all_inputs(bw_consumers, rnn_bw.output[rnn_output_index], slice_node_bw)
