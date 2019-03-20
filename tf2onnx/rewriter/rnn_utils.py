# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn_utils - rnn support
"""

from __future__ import unicode_literals

import logging
from enum import Enum
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher # pylint: disable=unused-import


# pylint: disable=invalid-name,unused-argument,missing-docstring


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.rnn_utils")


class REWRITER_RESULT(Enum):
    SKIP = 1
    OK = 2
    FAIL = 3


class RnnWeight:
    def __init__(self, node, np_val, np_dtype):
        self.node = node
        self.value = np_val
        self.dtype = np_dtype


class RnnWeights:
    def __init__(self, kernel, bias, forget_bias):
        self.kernel = kernel
        self.bias = bias
        self.forget_bias = forget_bias


class RnnProperties:
    def __init__(self):
        # RNN input who are outside of rnn scope
        self.input_node = None
        self.input_id = None
        self.var_initializers = {}

        self.onnx_input_ids = {}

        self.time_major = False
        self.x_input_id = None # used to serve lstm's 1st input
        self.input_size = None
        self.hidden_size = None
        self.batch_size_node = None # only for fill constant workaround

    def is_valid(self):
        if not self.input_node:
            log.debug("no input node found for current rnn, skip")
            return False
        log.debug("input node with port id %s", self.input_id)
        return True


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


grublockcell_pattern = OpTypePattern("GRUBlockCell", name="GRUBlockCell")


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


rnn_cell_patterns = {
    RNNUnitType.LSTMCell: lstmcell_pattern,
    RNNUnitType.LSTMBlockCell: lstmblockcell_pattern,
    RNNUnitType.GRUCell: grucell_pattern,
    RNNUnitType.GRUBlockCell: grublockcell_pattern
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
        log.debug("this should not be a dynamic_rnn loop, found_time: %s, is_rnn_out_ta: %s",
                  found_time, is_rnn_out_ta)
        return None

    if not loop_properties.tensor_array_inputs:
        log.debug("this should not be a dynamic_rnn loop, no ta input is found")
        return None

    return time_var, iteration_var


def get_weights_from_const_node(g, node):
    temp = node
    val = None
    dtype = None
    # this would help ignore Identity in non-const_folded graph.
    while temp.type == 'Identity':
        temp = temp.inputs[0]

    if temp and temp.type == 'Const':
        val = temp.get_tensor_value(as_list=False)
        dtype = utils.map_onnx_to_numpy_type(g.get_dtype(temp.output[0]))
        log.debug("found weights %s", temp.name)
    else:
        log.debug("weight node seems not to be Const, skip, node name is %s", temp.name)
        return None

    return RnnWeight(node, val, dtype)


def check_is_timemajor_transpose(node):
    # TensorFlow transpose node has perm as its second input
    if node.type != "Transpose":
        return False

    perm_node = node.inputs[1]
    if perm_node.is_const():
        return list(node.inputs[1].get_tensor_value(as_list=False)) == [1, 0, 2]
    if check_is_unfolded_perm(perm_node):
        return True
    raise ValueError("Not supported yet")


# todo: fix this
def check_is_unfolded_perm(perm_node):
    # For some case, like HallWay, the perm is a ConcatV2,
    # but it should be calculated when constant-fold. TODO: investigate why not constant fold.
    # current workaround: use np to calculate the val explicitly.
    if perm_node.type == "ConcatV2" and len(perm_node.inputs) == 3:
        const_node_val = perm_node.inputs[0].get_tensor_value(as_list=False)
        if list(const_node_val) != [1, 0]:
            return False

        range_node = perm_node.inputs[1]
        range_start = range_node.inputs[0].get_tensor_value()
        range_limit = range_node.inputs[1].get_tensor_value()
        range_delta = range_node.inputs[2].get_tensor_value()
        if range_node.type == "Range" and range_start == 2 and range_limit == 3 and range_delta == 1:
            # we just hard code this now
            # todo: refine this
            return True
    return False

def is_reverse_op(op):
    return op.type in ("ReverseV2", "ReverseSequence")


def is_concat_op(op):
    return op.type in ("Concat", "ConcatV2", "ConcatV3")


def is_tensor_array_scatter_op(op):
    return op.type in ("TensorArrayScatterV2", "TensorArrayScatterV3")


def is_tensor_array_gather_op(op):
    return op.type in ("TensorArrayGatherV2", "TensorArrayGatherV3")


def is_tensor_array_read_op(op):
    return op.type in ("TensorArrayReadV2", "TensorArrayReadV3")


def is_tensor_array_write_op(op):
    return op.type in ("TensorArrayWriteV2", "TensorArrayWriteV3")


def is_tensor_array_op(op):
    return op.type in ("TensorArrayV2", "TensorArrayV3")


def is_tensor_array_size_op(op):
    return op.type in ("TensorArraySizeV2", "TensorArraySizeV3")


def is_placeholder_op(op):
    return op.type == "Placeholder"


def is_loopcond_op(op):
    return op.type == "LoopCond"


def is_select_op(op):
    return op.type == "Select"


def is_slice_op(op):
    return op.type == "Slice"
