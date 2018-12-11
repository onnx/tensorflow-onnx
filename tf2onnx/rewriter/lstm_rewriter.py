# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.lstm_rewriter - lstm support
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.rewriter.rnn_utils import RnnWeights, RNNUnitType, get_weights_from_const_node, \
    is_tensor_array_write_op, is_tensor_array_op, is_tensor_array_gather_op, is_tensor_array_size_op, \
    check_is_timemajor_transpose

from tf2onnx.rewriter.unit_rewriter_base import UnitRewriterBase

# pylint: disable=invalid-name,unused-argument,missing-docstring

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.lstm_rewriter")


class LSTMUnitRewriter(UnitRewriterBase):
    def __init__(self, g):
        super(LSTMUnitRewriter, self).__init__(g)
        self.switch_checkers = {
            # True means we need parse its initial value in later logic.
            "ct": (self._ct_switch_check, self._connect_lstm_yc_to_graph, True),
            "ht": (self._ht_switch_check, self._connect_lstm_yh_to_graph, True),
            "ct_ht": (self._ct_ht_shared_switch_check, self._connect_lstm_ych_to_graph, True),
            "output": (self._output_switch_check, self._connect_lstm_output_to_graph, False),
        }

    def run(self): # FIXME: pylint: disable=arguments-differ
        return super(LSTMUnitRewriter, self).run(RNNUnitType.LSTMCell)

    def get_rnn_scope_name(self, match):
        # take the cell output and go up 3 levels to find the scope:
        # name of h is like root/while/lstm_cell/mul_2
        # root is the dynamic rnn's scope name.
        # root/while/lstm_cell is cell's scope name
        h_node = match.get_op("ht")
        parts = h_node.name.split('/')
        rnn_scope_name = '/'.join(parts[0:-3])
        return rnn_scope_name

    def get_weight_and_bias(self, match):
        # if one of them is not match, just return
        w_e = match.get_op("cell_kernel")
        w = get_weights_from_const_node(w_e)
        if not w:
            return None

        # check https://www.tensorflow.org/versions/r1.8/api_docs/cc/class/tensorflow/ops/bias-add
        # for bias_add data format
        bias_add = match.get_op("bias_add")
        if bias_add.data_format != "NHWC":
            log.debug("BiasAdd data_format is not NHWC, SKIP")
            return None

        b_e = match.get_op("cell_bias")
        b = get_weights_from_const_node(b_e)
        if not b or b.value.shape[0] != w.value.shape[1]:
            log.warning("cell_kernel and cell_bias's dimensions does not match, skip")
            return None

        ft_bias = match.get_op("ft_bias")
        ft = get_weights_from_const_node(ft_bias)
        if not ft:
            return None

        if not (len(ft.value) == 1 and b_e.dtype == ft_bias.dtype):
            return None

        return RnnWeights(w, b, ft)

    def _ct_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        # original we use c.inputs[0] == match.get_op("ft") to check c initializer for LSTMCell
        # but in BasicLSTMCell, c.inputs[1] is "ft", that's because BasicLSTMCell and LSTMCell's call
        # function are defining the multiplication with different order.
        # So we change to match.get_op("ft") in c.inputs
        mul_nodes = [c for c in identity_consumers if c.type == "Mul" and match.get_op("ft") in c.inputs]
        if len(mul_nodes) == 1:
            log.debug("find c initializer value at %s", enter_target_node_input_id)
            return enter_target_node_input_id
        log.debug("multiple Mul matching found, cannot identify c initializer")
        return None

    def _ht_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        concat_nodes = [c for c in identity_consumers if c == match.get_op("xh")]
        if len(concat_nodes) == 1:
            log.debug("find h initializer value at %s", enter_target_node_input_id)
            return enter_target_node_input_id
        log.debug("%d Concat matching found, cannot identify h initializer", len(concat_nodes))
        return None

    # when state is not tuple, ct and ht may share same switch.
    def _ct_ht_shared_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        slices = [c for c in identity_consumers if c.type == "Slice"]
        if not slices:
            log.debug("find no switch_identity_slice nodes")
            return None

        c_slice = None
        h_slice = None
        hidden_size = None
        for s in slices:
            slice_consumers = self.g.find_output_consumers(s.output[0])
            if len(slice_consumers) != 1:
                continue

            s_begin = s.inputs[1].get_tensor_value()
            s_size = s.inputs[2].get_tensor_value()
            hidden_size = s_size[1]
            if list(s_begin) == [0, 0]:
                c_slice = s
            elif list(s_begin) == [0, hidden_size]:
                h_slice = s

        if c_slice and h_slice:
            log.debug("find c_h shared initializer value at %s", enter_target_node_input_id)
            return enter_target_node_input_id
        return None

    def _output_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        ta_write_nodes = [c for c in identity_consumers if is_tensor_array_write_op(c)]
        if len(ta_write_nodes) == 1:
            enter_target_node = self.g.get_node_by_output(enter_target_node_input_id)
            if is_tensor_array_op(enter_target_node):
                log.debug("found output switch node")
                return enter_target_node_input_id
            log.debug("found enter target node is not ta node")
            return None
        log.debug("%d TensorArrayWrite matching found, cannot validate output switch", len(ta_write_nodes))
        return None

    def process_weights_and_bias(self, rnn_weights, rnn_props):
        w_r_icfo = rnn_weights.kernel.value
        w_dtype = rnn_weights.kernel.dtype
        b_r_icfo = rnn_weights.bias.value
        b_dtype = rnn_weights.bias.dtype
        ft_bias_scalar = rnn_weights.forget_bias.value

        # split bias for each hidden unit
        # b_r_icfo: (4 * num_units,)
        bias_dim = b_r_icfo.shape[0]
        hidden_size = int(bias_dim/4)
        b_r_icfo = np.reshape(b_r_icfo, (1, bias_dim))
        bias_gates = np.split(b_r_icfo, 4, axis=1)
        ft_bias = np.add(bias_gates[2], ft_bias_scalar[0])
        wb_bias_iofc = np.concatenate((bias_gates[0], bias_gates[3], ft_bias, bias_gates[1]), axis=1)

        # fill Rb with empty since in TF, we have only one bias.
        rb_bias_iofc = np.zeros((1, bias_dim), dtype=b_dtype)
        B = np.concatenate((wb_bias_iofc, rb_bias_iofc), axis=1)
        assert B.shape == (1, 2 * bias_dim)

        [wx, wh] = np.split(w_r_icfo, [-1 * hidden_size])
        input_size = wx.shape[0]
        assert wx.shape[0] == input_size
        assert int(wx.shape[1]/4) == hidden_size

        # split weight for gates
        w_gates = np.split(wx, 4, axis=1)
        new_wx = np.concatenate((w_gates[0], w_gates[3], w_gates[2], w_gates[1]), axis=1)

        h_gates = np.split(wh, 4, axis=1)
        new_wh = np.concatenate((h_gates[0], h_gates[3], h_gates[2], h_gates[1]), axis=1)
        W_iofc = np.transpose(new_wx)
        R_iofc = np.transpose(new_wh)

        W = np.array([W_iofc], w_dtype)
        R = np.array([R_iofc], w_dtype)

        # create node
        w_name = utils.make_name("W")
        w_node = self.g.make_const(w_name, W, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = self.g.make_const(r_name, R, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = self.g.make_const(b_name, B, skip_conversion=True)

        rnn_props.input_size = input_size
        rnn_props.hidden_size = hidden_size
        rnn_props.onnx_input_ids["W"] = w_node.output[0]
        rnn_props.onnx_input_ids["R"] = r_node.output[0]
        rnn_props.onnx_input_ids["B"] = b_node.output[0]

    def process_var_init_nodes(self, rnn_props):
        init_h_id = None
        init_c_id = None
        if "ct_ht" in rnn_props.var_initializers:
            init_h_id, init_c_id = self._process_non_tuple_ch_init_nodes(rnn_props)
        elif "ct" in rnn_props.var_initializers and "ht" in rnn_props.var_initializers:
            init_h_id, init_c_id = self._process_tuple_ch_init_nodes(rnn_props)
        else:
            raise ValueError("no initializers, unexpected")
        assert init_h_id and init_c_id
        rnn_props.onnx_input_ids["initial_h"] = init_h_id
        rnn_props.onnx_input_ids["initial_c"] = init_c_id

    def _process_non_tuple_ch_init_nodes(self, rnn_props):
        input_id = rnn_props.var_initializers["ct_ht"]
        hidden_size = rnn_props.hidden_size

        # todo: remove this once Fill ops is supported
        fill_ch_init_node = self._workaround_fill_ch_init_node(input_id, rnn_props)
        if fill_ch_init_node:
            return fill_ch_init_node.output[0], fill_ch_init_node.output[0]

        attr = {"axes": [1], "starts": [0], "ends": [hidden_size]}
        slice_node1 = self.g.make_node("Slice", [input_id], attr)
        unsqueeze_node_1 = self.g.make_node("Unsqueeze", [slice_node1.output[0]], attr={"axes": [0]})

        attr = {"axes": [1], "starts": [hidden_size], "ends": [hidden_size*2]}
        slice_node2 = self.g.make_node("Slice", [input_id], attr)
        unsqueeze_node_2 = self.g.make_node("Unsqueeze", [slice_node2.output[0]], attr={"axes": [0]})

        self.all_nodes.extend([slice_node1, slice_node2, unsqueeze_node_1, unsqueeze_node_2])
        return unsqueeze_node_1.output[0], unsqueeze_node_2.output[0]

    def _process_tuple_ch_init_nodes(self, rnn_props):
        h_init_input_id = rnn_props.var_initializers["ht"]
        c_init_input_id = rnn_props.var_initializers["ct"]
        h_node_output = self._process_c_or_h_init_nodes(h_init_input_id, rnn_props)
        c_node_output = self._process_c_or_h_init_nodes(c_init_input_id, rnn_props)
        return h_node_output, c_node_output

    def _process_c_or_h_init_nodes(self, initializer_input_id, rnn_props):
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

    def create_rnn_node(self, rnn_props):
        # specify if the RNN is forward, reverse, or bidirectional.
        # Must be one of forward (default), reverse, or bidirectional.
        # Here we won't mark bidirectional/reverse, we will have another rewriter running
        # after this one, which will based on patterns to combine a forward LSTM and a
        # backward LSTM into a bidirectional one.
        direction = "forward"
        num_direction = 1
        # todo: input_forget
        attr = {"direction": direction, "hidden_size": rnn_props.hidden_size}
        inputs = rnn_props.onnx_input_ids
        lstm_inputs = [
            inputs["X"], inputs["W"], inputs["R"], inputs["B"],
            inputs["sequence_lens"], inputs["initial_h"], inputs["initial_c"]]

        x_shape = self.g.get_shape(lstm_inputs[0])
        x_seq_length = x_shape[0]
        x_batch_size = x_shape[1]
        out_dtype = self.g.get_dtype(lstm_inputs[0])

        lstm_node = self.g.make_node("LSTM", lstm_inputs, attr=attr, output_count=3,
                                     shapes=[[x_seq_length, num_direction, x_batch_size, rnn_props.hidden_size],
                                             [num_direction, x_batch_size, rnn_props.hidden_size],
                                             [num_direction, x_batch_size, rnn_props.hidden_size]],
                                     dtypes=[out_dtype, out_dtype, out_dtype])
        return lstm_node

    def _connect_lstm_yh_to_graph(self, lstm_node, exit_node, rnn_props):
        # in tf, y_h output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        output_id = lstm_node.output[1]
        lstm_yh_shape = self.g.get_shape(output_id)
        squeeze_output_shape = [lstm_yh_shape[1], lstm_yh_shape[2]]
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [0]},
                                        shapes=[squeeze_output_shape])

        self.all_nodes.extend([squeeze_node])
        self.g.replace_all_inputs(self.all_nodes, exit_node.output[0], squeeze_node.output[0])

    def _connect_lstm_yc_to_graph(self, lstm_node, exit_node, rnn_props):
        # in tf, y_c output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        output_id = lstm_node.output[2]
        lstm_yc_shape = self.g.get_shape(output_id)
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [0]},
                                        shapes=[[lstm_yc_shape[1], lstm_yc_shape[2]]],
                                        dtypes=[self.g.get_dtype(output_id)])
        self.all_nodes.extend([squeeze_node])
        self.g.replace_all_inputs(self.all_nodes, exit_node.output[0], squeeze_node.output[0])

    def _connect_lstm_ych_to_graph(self, lstm_node, exit_node, rnn_props):
        # in tf, concat of y_c and y_h output shape is: [batch, hidden *2]
        # in onnx, y_c/y_h output shape is: [number_directions, batch, hidden]
        yc_shape = self.g.get_shape(lstm_node.output[2])
        concat_output_shape = [yc_shape[0], yc_shape[1], yc_shape[2] * 2]
        concat = self.g.make_node("Concat", [lstm_node.output[2], lstm_node.output[1]],
                                  attr={"axis": 2}, shapes=[concat_output_shape],
                                  dtypes=[self.g.get_dtype(lstm_node.output[2])])

        squeeze_output_shape = [concat_output_shape[1], concat_output_shape[2]]
        squeeze_node = self.g.make_node("Squeeze", [concat.output[0]], attr={"axes": [0]},
                                        shapes=[squeeze_output_shape],
                                        dtypes=[self.g.get_dtype(concat.output[0])])

        self.all_nodes.extend([concat, squeeze_node])
        self.g.replace_all_inputs(self.all_nodes, exit_node.output[0], squeeze_node.output[0])

    def _connect_lstm_output_to_graph(self, lstm_node, exit_node, rnn_props):
        exit_consumers = self.g.find_output_consumers(exit_node.output[0])
        gather_node = self._validate_output_exit_consumers(exit_consumers)
        if len(exit_consumers) != 2 or not gather_node:
            log.debug("lstm output exit node has %d consumers", len(exit_consumers))
            raise ValueError("lstm output exit node check failed")

        # gather output for sure has shape [time, batch, hidden]
        gather_output_id = gather_node.output[0]
        log.debug("found output ta gather node %s", gather_output_id)
        # in tf batch major mode, output shape is : [batch, time, hidden]
        # in time major mode, output shape is: [time, batch, hidden]
        # in onnx, output shape is : [time, num_directions, batch, hidden]

        output_id = lstm_node.output[0]
        lstm_output_shape = self.g.get_shape(output_id)
        squeeze_output_shape = [lstm_output_shape[0], lstm_output_shape[2], lstm_output_shape[3]]
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [1]},
                                        shapes=[squeeze_output_shape],
                                        dtypes=[self.g.get_dtype(output_id)])

        if not rnn_props.time_major:
            gather_consumers = self.g.find_output_consumers(gather_output_id)
            gather_trans_consumers = [n for n in gather_consumers if check_is_timemajor_transpose(n)]
            if len(gather_trans_consumers) != 1:
                raise ValueError("batch major should expect a transpose after gather")
            trans = gather_trans_consumers[0] # trans has rnn scope name

            # we just check the transpose here, but will not re-use it, because
            # it may hold non-const perms. so we re-create a new transpose to replace it
            attr = {"perm": np.array([1, 0, 2], dtype=np.int64)}
            trans_output_shape = [squeeze_output_shape[1], squeeze_output_shape[0], squeeze_output_shape[2]]
            new_trans = self.g.make_node("Transpose", [squeeze_node.output[0]], attr,
                                         shapes=[trans_output_shape],
                                         dtypes=[self.g.get_dtype(squeeze_node.output[0])])

            self.g.replace_all_inputs(self.all_nodes, trans.output[0], new_trans.output[0])
            self.all_nodes.extend([new_trans])

        self.g.replace_all_inputs(self.all_nodes, gather_output_id, squeeze_node.output[0])
        self.all_nodes.extend([squeeze_node])

    def _validate_output_exit_consumers(self, exit_consumers):
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
