# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.gruBlock_rewriter - gruBlock support
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from tf2onnx import utils
from tf2onnx.rewriter.gru_rewriter import GRUUnitRewriter
from tf2onnx.rewriter.rnn_utils import RNNUnitType, get_weights_from_const_node, \
    is_tensor_array_read_op, is_tensor_array_scatter_op


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.grublock_rewriter")

# pylint: disable=invalid-name,unused-argument,missing-docstring

class GRUBlockUnitRewriter(GRUUnitRewriter):
    def __init__(self, g):
        super(GRUBlockUnitRewriter, self).__init__(g)
        self.switch_checkers = {
            # True means we need parse its initial value in later logic.
            # in tensorflow, switch is a good op that we can use to trace other ops that needed
            "state": (self._state_switch_check, self._connect_gru_state_to_graph, True),
            "output": (self._output_switch_check, self._connect_gru_output_to_graph, False),
        }

    def run(self):
        return super(GRUBlockUnitRewriter, self).run_with_unit_type(RNNUnitType.GRUBlockCell)

    def get_rnn_scope_name(self, match):
        # take the cell output and go up 3 levels to find the scope:
        # name of h is like root/while/gru_cell/mul_2
        # root is the dynamic rnn's scope name.
        # root/while/gru_cell is cell's scope name
        h_node = match.get_op("GRUBlockCell").inputs[0]
        parts = h_node.name.split('/')
        rnn_scope_name = '/'.join(parts[0:-2])
        return rnn_scope_name

    def get_cell_scope_name(self, match):
        cell_node = match.get_op("GRUBlockCell")
        return cell_node.name[:cell_node.name.rfind("/")]

    def get_weight_and_bias(self, match):

        node = match.get_op("GRUBlockCell")
        # from tf, it can be known that, the inputs index and meaning of input data is:
        # 0-input, 1-state, 2-gate_kernel, 3-hidden_kernel, 4-gate_bias, 5-hidden_bias
        gate_kernel = get_weights_from_const_node(node.inputs[2].inputs[0])
        gate_bias = get_weights_from_const_node(node.inputs[4].inputs[0])
        hidden_kernel = get_weights_from_const_node(node.inputs[3].inputs[0])
        hidden_bias = get_weights_from_const_node(node.inputs[5].inputs[0])
        if not all([gate_kernel, gate_bias, hidden_kernel, hidden_bias]):
            log.debug("rnn weights check failed, skip")
            return None
        log.debug("find needed weights")
        res = {"gate_kernel": gate_kernel,
               "gate_bias": gate_bias,
               "hidden_kernel": hidden_kernel,
               "hidden_bias": hidden_bias}
        return res

    def find_inputs(self, rnn_scope_name, rnn_props, match, input_blacklist=None):
        cell_node = match.get_op("GRUBlockCell")
        read_node = cell_node.inputs[0]
        utils.make_sure(is_tensor_array_read_op(read_node), "ta read op check fail")
        enter_node = read_node.inputs[2]
        utils.make_sure(enter_node.type == "Enter", "enter check fail")
        scatter_node = enter_node.inputs[0]
        utils.make_sure(is_tensor_array_scatter_op(scatter_node), "ta scatter check fail")

        node = scatter_node.inputs[2]
        node_id = scatter_node.input[2]
        # dynamic_rnn may insert transpose op if input data format is [B, T, D]
        if node.type == "Transpose" and node.name.startswith(rnn_scope_name):
            node_id = node.input[0]
            node = node.inputs[0]

        utils.make_sure(not node.name.startswith(rnn_scope_name), "rnn input should not has rnn scope name")
        rnn_props.input_node = node
        rnn_props.input_id = node_id

    @staticmethod
    def _state_switch_check(enter_target_node_input_id, identity_consumers, match):
        node = match.get_op("GRUBlockCell")
        if node == identity_consumers[0]:
            log.debug("find state initializer value at %s", enter_target_node_input_id)
            return enter_target_node_input_id
        return None

    @staticmethod
    def get_rnn_activation(match):
        return "Tanh"

    def create_rnn_node(self, rnn_props):
        # specify if the RNN is forward, reverse, or bidirectional.
        # Must be one of forward (default), reverse, or bidirectional.
        # Here we won't mark bidirectional/reverse, we will have another rewriter running after this one,
        # which will based on patterns to combine a forward GRU and a backward GRU into a bidirectional one.
        direction = "forward"
        num_direction = 1
        # todo: input_forget
        attr = {"direction": direction, "hidden_size": rnn_props.hidden_size,
                "activations": ["sigmoid", rnn_props.activation]}
        inputs = rnn_props.onnx_input_ids
        gru_inputs = [
            inputs["X"], inputs["W"], inputs["R"], inputs["B"],
            inputs["sequence_lens"], inputs["initial_state"]]

        x_shape = self.g.get_shape(gru_inputs[0])
        x_dtype = self.g.get_dtype(gru_inputs[0])
        x_seq_length = x_shape[0]
        x_batch_size = x_shape[1]
        gru_node = self.g.make_node("GRU", gru_inputs, attr=attr, output_count=2,
                                    shapes=[[x_seq_length, num_direction, x_batch_size, rnn_props.hidden_size],
                                            [num_direction, x_batch_size, rnn_props.hidden_size]],
                                    dtypes=[x_dtype, x_dtype])
        return gru_node
