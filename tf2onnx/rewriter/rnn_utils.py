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


class RnnInitializers:
    def __init__(self, c_init, h_init, c_h_shared_init):
        self.c_init_input_id = None
        self.h_init_input_id = None
        self.share_init_node = None
        self.share_init_input_id = None

        if c_h_shared_init:
            self.share_init_input_id = c_h_shared_init
            self.share_init_node = True
        else:
            self.c_init_input_id = c_init
            self.h_init_input_id = h_init
            self.share_init_node = False


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
                OpTypePattern("Mul", inputs=[
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


class RNNUnitType(Enum):
    LSTMCell = 0  # TF LSTMCell and BasicLSTMCell share the same pattern
    GRUCell = 1
    GRUBlockCell = 2


# describe the body graph's input and output node
class SubGraphMetadata(object):
    def __init__(self, g, input_ids, output_ids, initial_input_ids):
        self.g = g
        self.input_ids = input_ids
        self.output_ids = output_ids

        self.initial_input_ids = initial_input_ids

        # sub-graph boundary
        self.other_enter_input_ids = []


class BodyGraphDict():
    BODY_GRAPH_DICT = {}

    def __init__(self, g):
        self.g = g

    @staticmethod
    def add_body_graph_info(body_owner_name, body_graph):
        if body_owner_name not in BodyGraphDict.BODY_GRAPH_DICT:
            BodyGraphDict.BODY_GRAPH_DICT[body_owner_name] = body_graph
        else:
            raise ValueError("body_owner_name " + body_owner_name + " already exists as a key")

    @staticmethod
    def pop_body_graph_info(body_owner_name):
        val = BodyGraphDict.BODY_GRAPH_DICT[body_owner_name]
        del BodyGraphDict.BODY_GRAPH_DICT[body_owner_name]
        return val

    @staticmethod
    def has_body_graph_info(body_owner_name):
        return body_owner_name in BodyGraphDict.BODY_GRAPH_DICT

    @staticmethod
    def get_body_graph_output_names():
        output_names = []
        for k in BodyGraphDict.BODY_GRAPH_DICT:
            _output_names = BodyGraphDict.BODY_GRAPH_DICT[k].output_ids
            output_names.extend(_output_names)
        return set(output_names)


rnn_cell_patterns = {
    RNNUnitType.LSTMCell: lstmcell_pattern,
    RNNUnitType.GRUCell: grucell_pattern,
    RNNUnitType.GRUBlockCell: grublockcell_pattern
}


def get_pattern(cell_type_name):
    return rnn_cell_patterns[cell_type_name]


def get_weights_from_const_node(node):
    temp = node
    val = None
    dtype = None
    # this would help ignore Identity in non-const_folded graph.
    while temp.type == 'Identity':
        temp = temp.inputs[0]

    if temp and temp.type == 'Const':
        val = temp.get_tensor_value()
        dtype = utils.ONNX_TO_NUMPY_DTYPE[temp.dtype]
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
        return list(node.inputs[1].get_tensor_value()) == [1, 0, 2]
    if check_is_unfolded_perm(perm_node):
        return True
    raise ValueError("Not supported yet")


# todo: fix this
def check_is_unfolded_perm(perm_node):
    # For some case, like HallWay, the perm is a ConcatV2,
    # but it should be calculated when constant-fold. TODO: investigate why not constant fold.
    # current workaround: use np to calculate the val explicitly.
    if perm_node.type == "ConcatV2" and len(perm_node.inputs) == 3:
        const_node_val = perm_node.inputs[0].get_tensor_value()
        if list(const_node_val) != [1, 0]:
            return False

        range_node = perm_node.inputs[1]
        range_start = range_node.inputs[0].get_tensor_value()
        range_limit = range_node.inputs[1].get_tensor_value()
        range_delta = range_node.inputs[2].get_tensor_value()
        if range_node.type == "Range" and range_start == [2] and range_limit == [3] and range_delta == [1]:
            # we just hard code this now
            # todo: refine this
            return True
    return False


def is_reverse_op(op):
    return op.type in ("ReverseV2", "ReverseSequence")


def is_concat_op(op):
    return op.type in ("ConcatV2", "ConcatV3")


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
