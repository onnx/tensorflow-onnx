# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.lstm_rewriter - lstm support
"""

from __future__ import division
from __future__ import print_function

import collections
import logging
import numpy as np
import tf2onnx
from onnx import helper, defs, numpy_helper, checker, onnx_pb
from onnx import AttributeProto, TensorProto, GraphProto
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import *
from tf2onnx.rewriter.rnn_utils import *
from tf2onnx.rewriter.unit_rewriter_base import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.lstm_rewriter")


class LSTMUnitRewriter(UnitRewriterBase):
    def __init__(self, g):
        super(LSTMUnitRewriter, self).__init__(g)

    def run(self):
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
            return

        # check https://www.tensorflow.org/versions/r1.8/api_docs/cc/class/tensorflow/ops/bias-add
        # for bias_add data format
        bias_add = match.get_op("bias_add")
        if bias_add.data_format != "NHWC":
            log.debug("BiasAdd data_format is not NHWC, SKIP")
            return

        b_e = match.get_op("cell_bias")
        b = get_weights_from_const_node(b_e)
        if not b or b.value.shape[0] != w.value.shape[1]:
            log.warning("cell_kernel and cell_bias's dimentions does not match, skip")
            return 

        ft_bias = match.get_op("ft_bias")
        ft = get_weights_from_const_node(ft_bias)
        if not ft:
            return

        if not (len(ft.value) == 1 and b_e.dtype == ft_bias.dtype):
            return

        return RnnWeights(w, b, ft)

    def ct_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        # original we use c.inputs[0] == match.get_op("ft") to check c initilier for LSTMCell
        # but in BasicLSTMCell, c.inputs[1] is "ft", that's because BasicLSTMCell and LSTMCell's call 
        # function are defining the multiplication with different order. So we change to match.get_op("ft") in c.inputs
        mul_nodes = [c for c in identity_consumers if c.type == "Mul" and match.get_op("ft") in c.inputs]
        if len(mul_nodes) == 1:
            log.info("find c initializer value at " + enter_target_node_input_id)
            return enter_target_node_input_id
        elif len(mul_nodes) > 1:
            raise ValueError("multiple Mul matching found, cannot identify c initializer")

    def ht_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        concat_nodes = [c for c in identity_consumers if c == match.get_op("xh")]
        if len(concat_nodes) == 1:
            log.info("find h initializer value at " + enter_target_node_input_id)
            return enter_target_node_input_id
        elif len(concat_nodes) > 1:
            raise ValueError(str(len(concat_nodes)) + "Concat matching found, cannot identify h initializer")

    def ct_ht_shared_switch_check(self, enter_target_node_input_id, identity_consumers, match):
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
            return enter_target_node_input_id

    def process_weights_and_bias(self, rnn_weights):
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

        return W, R, B, input_size, hidden_size
