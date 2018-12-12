# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.gru_rewriter - gru support
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np

from tf2onnx import utils
from tf2onnx.rewriter.unit_rewriter_base import UnitRewriterBase
from tf2onnx.rewriter.rnn_utils import get_weights_from_const_node, is_tensor_array_read_op, \
    is_tensor_array_scatter_op, is_tensor_array_write_op, is_tensor_array_op, is_tensor_array_gather_op, \
    is_tensor_array_size_op, check_is_timemajor_transpose, RNNUnitType

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.gru_rewriter")

# pylint: disable=invalid-name,unused-argument,missing-docstring

class GRUUnitRewriter(UnitRewriterBase):
    def __init__(self, g):
        super(GRUUnitRewriter, self).__init__(g)
        self.switch_checkers = {
            # True means we need parse its initial value in later logic.
            # in tensorflow, switch is a good op that we can use to trace other ops that needed
            "state": (self._state_switch_check, self._connect_gru_state_to_graph, True),
            "output": (self._output_switch_check, self._connect_gru_output_to_graph, False),
        }

    def run(self): # FIXME: pylint: disable=arguments-differ
        return super(GRUUnitRewriter, self).run(RNNUnitType.GRUCell)

    def run_with_unit_type(self, unit_type):
        return super(GRUUnitRewriter, self).run(unit_type)

    def get_rnn_scope_name(self, match):
        # take the cell output and go up 3 levels to find the scope:
        # name of h is like root/while/gru_cell/mul_2
        # root is the dynamic rnn's scope name.
        # root/while/gru_cell is cell's scope name
        concat_node = match.get_op("cell_inputs")
        read_node = concat_node.inputs[0]
        parts = read_node.name.split('/')
        rnn_scope_name = '/'.join(parts[0:-2])
        return rnn_scope_name

    def get_cell_scope_name(self, match):
        """
        when name of rnn scope is rnnxx, then cell scope name will be rnnxx/while/grucell

        when bi-directional rnn use only one cell, then each single rnn will weights and computation graph
        this means that these shared nodes will have same rnn scope name,
        so keeps nodes under cell scope name when call "run_single_match", and delete these nodes in "run"
        """
        # this cell_node is concat, its name will something like rnnxx/while/grucell/concat_n
        cell_node = match.get_op("cell_inputs")
        return cell_node.name[:cell_node.name.rfind("/")]

    def get_weight_and_bias(self, match):
        gate_kernel = get_weights_from_const_node(match.get_op("gate_kernel"))
        gate_bias = get_weights_from_const_node(match.get_op("gate_bias"))
        hidden_kernel = get_weights_from_const_node(match.get_op("hidden_kernel"))
        hidden_bias = get_weights_from_const_node(match.get_op("hidden_bias"))
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
        concat_node = match.get_op("cell_inputs")
        read_node = concat_node.inputs[0]
        utils.make_sure(is_tensor_array_read_op(read_node), "ta read check fail")
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

        assert not node.name.startswith(rnn_scope_name)
        rnn_props.input_node = node
        rnn_props.input_id = node_id

    @staticmethod
    def _state_switch_check(enter_target_node_input_id, identity_consumers, match):
        concat_nodes = [c for c in identity_consumers if c == match.get_op("cell_inputs")]
        if len(concat_nodes) == 1:
            log.debug("find state initializer value at %s", enter_target_node_input_id)
            return enter_target_node_input_id
        log.debug("%d Concat matching found, cannot identify state initializer", len(concat_nodes))
        return None

    def _connect_gru_state_to_graph(self, gru_node, exit_node, rnn_props):
        # in tf, state output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        output_id = gru_node.output[1]
        gru_state_shape = self.g.get_shape(output_id)
        output_shape = [gru_state_shape[1], gru_state_shape[2]]
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [0]},
                                        shapes=[output_shape], dtypes=[self.g.get_dtype(output_id)])

        self.all_nodes.extend([squeeze_node])
        self.g.replace_all_inputs(self.all_nodes, exit_node.output[0], squeeze_node.output[0])

    def _output_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        ta_write_nodes = [c for c in identity_consumers if is_tensor_array_write_op(c)]
        if len(ta_write_nodes) == 1:
            enter_target_node = self.g.get_node_by_output(enter_target_node_input_id)
            if is_tensor_array_op(enter_target_node):
                log.debug("found output switch node")
                return enter_target_node_input_id
            log.debug("found enter target node is not ta node")
            return None
        log.debug("%d TensorArrayWriteV3 matching found, cannot validate output switch", len(ta_write_nodes))
        return None

    def _connect_gru_output_to_graph(self, gru_node, exit_node, rnn_props):
        exit_consumers = self.g.find_output_consumers(exit_node.output[0])
        gather_node = self._validate_output_exit_consumers(exit_consumers)
        if len(exit_consumers) != 2 or not gather_node:
            log.debug("gru output exit node has %d consumers", len(exit_consumers))
            raise ValueError("gru output exit node check failed")

        # gather output for sure has shape [time, batch, hidden]
        gather_output_id = gather_node.output[0]
        log.debug("found output ta gather node %d", gather_output_id)
        # in tf batch major mode, output shape is : [batch, time, hidden]
        # in time major mode, output shape is: [time, batch, hidden]
        # in onnx, output shape is : [time, num_directions, batch, hidden]

        output_id = gru_node.output[0]
        gru_output_shape = self.g.get_shape(output_id)
        squeeze_output_shape = [gru_output_shape[0], gru_output_shape[2], gru_output_shape[3]]
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [1]},
                                        shapes=[squeeze_output_shape],
                                        dtypes=[self.g.get_dtype(output_id)])

        if not rnn_props.time_major:
            gather_consumers = self.g.find_output_consumers(gather_output_id)
            gather_trans_consumers = [n for n in gather_consumers if check_is_timemajor_transpose(n)]
            if len(gather_trans_consumers) != 1:
                raise ValueError("batch major should expect a transpose after gather")
            trans = gather_trans_consumers[0]  # trans has rnn scope name

            # we just check the transpose here, but will not re-use it, because
            # it may hold non-const perms. so we re-create a new transpose to replace it
            attr = {"perm": np.array([1, 0, 2], dtype=np.int64)}
            trans_input_shape = self.g.get_shape(squeeze_node.output[0])
            trans_output_shape = [trans_input_shape[1], trans_input_shape[0], trans_input_shape[2]]
            new_trans = self.g.make_node("Transpose", [squeeze_node.output[0]], attr=attr,
                                         shapes=[trans_output_shape],
                                         dtypes=[self.g.get_dtype(squeeze_node.output[0])])
            self.g.replace_all_inputs(self.all_nodes, trans.output[0], new_trans.output[0])
            self.all_nodes.extend([new_trans])

        self.g.replace_all_inputs(self.all_nodes, gather_output_id, squeeze_node.output[0])
        self.all_nodes.extend([squeeze_node])

    @staticmethod
    def _validate_output_exit_consumers(exit_consumers):
        if len(exit_consumers) != 2:
            return None

        gather_node = None
        for n in exit_consumers:
            if is_tensor_array_gather_op(n):
                gather_node = n
            elif is_tensor_array_size_op(n):
                continue
            else:
                return None
        return gather_node

    def get_rnn_input_blacklist(self, rnn_weights, rnn_props):
        var_init_nodes = []
        for _, init_input_id in rnn_props.var_initializers.items():
            init_node = self.g.get_node_by_output(init_input_id)
            var_init_nodes.append(init_node)
        blacklist_inputs = []
        blacklist_inputs.extend(var_init_nodes)
        # weight/bias inputs, and c/h initializer are dynamic_rnn/GRUCell's parameters.
        # we will use them to filter out the dynamic_rnn's input tensor.
        for _, value in rnn_weights.items():
            blacklist_inputs.append(value.node)

        return blacklist_inputs

    def process_weights_and_bias(self, rnn_weights, rnn_props):
        """
        why split the data in this way should refer to code of tensorflow GRU cell and official document of ONNX GRU
        """
        # from code of tensorflow GRU cell, it can be known that shape of hidden_kernel(or candidate_kernel)
        # is (input_size+hidden_unit, hidden_unit)
        hidden_size = rnn_weights["hidden_kernel"].value.shape[1]
        input_size = rnn_weights["hidden_kernel"].value.shape[0] - hidden_size
        weight_dtype = rnn_weights["hidden_kernel"].dtype
        bias_dtype = rnn_weights["hidden_bias"].dtype
        # below code will use same notation as ONNX document
        # z means update gate, r means reset gate, h means hidden gate;
        # at this time weights of gate include input and state, will split it next
        r_kernel, z_kernel = np.split(rnn_weights["gate_kernel"].value, [hidden_size], axis=1)
        h_kernel = rnn_weights["hidden_kernel"].value
        r_bias, z_bias = np.split(rnn_weights["gate_bias"].value, [hidden_size], axis=0)
        h_bias = rnn_weights["hidden_bias"].value
        # ONNX GRU split weights of input and state, so have to split *_kernel
        input_r_kernel, state_r_kernel = np.split(r_kernel, [input_size], axis=0)
        input_z_kernel, state_z_kernel = np.split(z_kernel, [input_size], axis=0)
        input_h_kernel, state_h_kernel = np.split(h_kernel, [input_size], axis=0)
        W_zrh = np.concatenate((input_z_kernel, input_r_kernel, input_h_kernel), axis=1)
        R_zrh = np.concatenate((state_z_kernel, state_r_kernel, state_h_kernel), axis=1)
        # transpose weight matrix
        W_zrh = np.transpose(np.expand_dims(W_zrh, axis=0), axes=(0, 2, 1))
        R_zrh = np.transpose(np.expand_dims(R_zrh, axis=0), axes=(0, 2, 1))
        W_zrh = W_zrh.astype(weight_dtype)
        R_zrh = R_zrh.astype(weight_dtype)
        assert W_zrh.shape == (1, 3*hidden_size, input_size)
        assert R_zrh.shape == (1, 3*hidden_size, hidden_size)
        Wb_zrh = np.concatenate((z_bias, r_bias, h_bias), axis=0)
        # tf don't have bias for state, so use 0 instead
        zero = np.zeros_like(z_bias)
        Rb_zrh = np.concatenate((zero, zero, zero), axis=0)
        B_zrh = np.concatenate((Wb_zrh, Rb_zrh), axis=0)
        B_zrh = np.expand_dims(B_zrh, axis=0)
        B_zrh = B_zrh.astype(bias_dtype)
        assert B_zrh.shape == (1, 6*hidden_size)
        # create const ONNX node
        w_name = utils.make_name("W")
        w_node = self.g.make_const(w_name, W_zrh, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = self.g.make_const(r_name, R_zrh, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = self.g.make_const(b_name, B_zrh, skip_conversion=True)

        rnn_props.input_size = input_size
        rnn_props.hidden_size = hidden_size
        rnn_props.onnx_input_ids["W"] = w_node.output[0]
        rnn_props.onnx_input_ids["R"] = r_node.output[0]
        rnn_props.onnx_input_ids["B"] = b_node.output[0]

    def process_var_init_nodes(self, rnn_props):
        assert "state" in rnn_props.var_initializers.keys()
        found_node_id = rnn_props.var_initializers["state"]
        init_state_id = self._process_init_nodes(found_node_id, rnn_props)
        rnn_props.onnx_input_ids["initial_state"] = init_state_id

    def _process_init_nodes(self, initializer_input_id, rnn_props):
        # copy from  lstm_rewriter
        # todo: remove this once Fill ops is supported
        fill_ch_init_node = self._workaround_fill_ch_init_node(initializer_input_id, rnn_props)
        if fill_ch_init_node:
            return fill_ch_init_node.output[0]

        node = self.g.get_node_by_output(initializer_input_id)
        if node.is_const():
            val = node.get_tensor_value()
            initial_name = utils.make_name("Const")
            new_val = np.expand_dims(val, axis=0)
            const_node = self.g.make_const(initial_name, new_val)
            return const_node.output[0]

        squeeze_node = self.g.make_node("Unsqueeze", [initializer_input_id], attr={"axes": [0]})
        self.g.replace_all_inputs(self.g.get_nodes(), initializer_input_id, squeeze_node.output[0])
        self.all_nodes.append(squeeze_node)
        return squeeze_node.output[0]

    @staticmethod
    def get_rnn_activation(match):
        # in tf, only activation of hidden gate is optional, input and update gate always use sigmoid
        activation_op = match.get_op("optional_activation")
        return activation_op.type

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
        x_seq_length = x_shape[0]
        x_batch_size = x_shape[1]
        out_dtype = self.g.get_dtype(gru_inputs[0])
        gru_node = self.g.make_node("GRU", gru_inputs, attr=attr, output_count=2,
                                    shapes=[[x_seq_length, num_direction, x_batch_size, rnn_props.hidden_size],
                                            [num_direction, x_batch_size, rnn_props.hidden_size]],
                                    dtypes=[out_dtype, out_dtype])
        return gru_node
