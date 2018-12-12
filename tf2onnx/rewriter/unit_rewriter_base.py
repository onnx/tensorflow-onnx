# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn_unit_base - lstm support
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from onnx import onnx_pb
from tf2onnx import utils
from tf2onnx.rewriter.rnn_utils import get_pattern, RnnProperties, \
     check_is_timemajor_transpose, REWRITER_RESULT
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher # pylint: disable=unused-import


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.rnn_unit_writer_base")

# pylint: disable=invalid-name,unused-argument,missing-docstring

# dynamic_rnn or bidirectional_dynamic_rnn related logic will be mapped to this base class.
class UnitRewriterBase(object):
    def __init__(self, g):
        self.g = g
        self.all_nodes = self.g.get_nodes()
        # checker signature : func_name(enter_target_node_input_id, identity_consumers, match)
        # exit connector signature: func_name(rnn_node, exit_node, rnn_props)
        self.switch_checkers = {}

    def run(self, unit_type):
        """
        main procedures:
        1 use cell op pattern to find cell >> the found cell is the start pointer of the procedures below
        2 find needed info from tensorflow graph:
            1 rnn scope name
            2 input_x
            3 weight
            4 sequence node
            5 initializer
            6 state output & hidden output
        3 process found info according to ONNX requirement

        remember: op pattern and scope name are useful
                  they are used to get needed info from tensorflow graph
                  raw found info need to be formatted according to ONNX requirement
        """
        # allow_reorder must be true. because LSTMCell and BasicLSTMCell's call function
        # are defining the calculation with different orders. Then we can share the same
        # pattern.
        cell_pattern = get_pattern(unit_type)
        matcher = GraphMatcher(cell_pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(self.g.get_nodes()))

        if match_results:
            for match in match_results:
                self.run_single_match(match)

            self.g.delete_unused_nodes(self.g.output_names)
            self.g.update_proto()
            self.print_step("finish handling")

        return self.g.get_nodes()

    def run_single_match(self, match):
        """
        methods to get needed info from tf graph:
          1 input_x: specific node in found cell, then trace TensorArrayReadV >..>input of "TensorArrayScatterV",
              if "Transpose" found under rnn scope, then input of "Transpose" is "input_x"
          2 weight: specific node in cell computation graph and specific op  pattern as input_x
          3 sequence node: "Identity" op with name "sequence_length", the name is hard code in tensorflow code
          4 state initializer: "LoopCond" and then specific op pattern >> LoopCond > Switch > Switch usage checker
          5 hidden output and state output: find switch and use switch checker to distinguish different switch nodes

          6 scope name of rnn and gru/lstm cell: specific node in cell computation graph,
              and use found convention in tensorflow code to split name of node to get needed scooe name

          most found info is stored in "rnn_props"
        """
        log.debug("=========================")
        self.print_step("start handling a new potential rnn cell")
        self.all_nodes = self.g.get_nodes()
        # FIXME:
        # pylint: disable=assignment-from-none,assignment-from-no-return

        # when bi-directional, node in while will be rnnxx/fw/fw/while/... >> scope name is rnnxx/fw/fw
        # when single direction, node in while will be rnnxx/while/... >> scope name is rnnxx
        # and rnnxx can be assigned by users but not "fw", though maybe "FW" in another tf version
        rnn_scope_name = self.get_rnn_scope_name(match)
        if not rnn_scope_name:
            log.debug("unable to find rnn scope name, skip")
            return REWRITER_RESULT.SKIP
        log.debug("rnn scope name is %s", rnn_scope_name)

        self.print_step("get_weight_and_bias starts")
        rnn_weights = self.get_weight_and_bias(match)
        if not rnn_weights:
            log.debug("rnn weights check failed, skip")
            return REWRITER_RESULT.SKIP

        rnn_props = RnnProperties()
        res = self.get_var_initializers(match, rnn_props, rnn_scope_name)
        if not res or not rnn_props.var_initializers.keys:
            log.debug("no cell variable initializers found, skip")
            return REWRITER_RESULT.SKIP

        seq_len_input_node = self.find_sequence_length_node(rnn_scope_name)
        input_filter = self.get_rnn_input_blacklist(rnn_weights, rnn_props)
        if seq_len_input_node:
            input_filter.append(seq_len_input_node)

        self.find_inputs(rnn_scope_name, rnn_props, match, input_filter)
        if not rnn_props.is_valid():
            log.debug("rnn properties are not valid, skip")
            return REWRITER_RESULT.SKIP

        if not self.process_input_x(rnn_props, rnn_scope_name):
            log.debug("rnn input x not found, skip")
            return REWRITER_RESULT.SKIP

        self.print_step("process the weights/bias/ft_bias, to fit onnx weights/bias requirements")
        self.process_weights_and_bias(rnn_weights, rnn_props)

        _, batch_size_node = self.process_seq_length(rnn_props, seq_len_input_node)
        rnn_props.batch_size_node = batch_size_node

        self.process_var_init_nodes(rnn_props)

        self.print_step("start to build new rnn node")

        rnn_props.activation = self.get_rnn_activation(match)

        rnn_node = self.create_rnn_node(rnn_props)
        self.all_nodes.append(rnn_node)

        self.print_step("start to handle outputs")
        # format of ONNX output is different with tf
        self.process_outputs(match, rnn_node, rnn_props, rnn_scope_name)
        # FIXME:
        # pylint: enable=assignment-from-none,assignment-from-no-return
        return REWRITER_RESULT.OK

# find needed info from graph
    def get_rnn_scope_name(self, match):
        pass

    def get_cell_scope_name(self, match):
        return None

    @staticmethod
    def get_rnn_activation(match):
        return None

    def get_weight_and_bias(self, match):
        pass

    def get_var_initializers(self, match, rnn_props, rnn_scope_name):
        """
        initializer op can be found by tracing from switch mode. while rnn has multiple switch nodes,
        so have to discriminate them by a check.
        switch nodes can be found by tracing LoopCond
        """
        loop_cond_op = None
        for n in self.g.get_nodes():
            if n.type == 'LoopCond' and n.name.startswith(rnn_scope_name):
                if not loop_cond_op:
                    loop_cond_op = n
                else:
                    log.debug("only a LoopCond is expected, rnn scope name:%s", rnn_scope_name)
                    return None

        if loop_cond_op is None:
            log.debug("No LoopCond op is found, skip")
            return None

        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for n in switch_nodes:
            if n.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")

            for var_name, funcs in self.switch_checkers.items():
                var_checker = funcs[0]
                if not funcs[2]:
                    continue
                enter_target_input_id = self.check_switch_by_usage_pattern(n, match, var_checker)
                if enter_target_input_id:
                    log.debug("found initializer node for " + var_name + ": " + enter_target_input_id)
                    rnn_props.var_initializers[var_name] = enter_target_input_id
                    break
        return rnn_props.var_initializers

    def find_sequence_length_node(self, rnn_scope_name):
        # "sequence_length" under current rnn scope is the seq len node (if there is).
        # this is hardcoded in dynamic_rnn().

        seq_len_nodes = []
        for n in self.g.get_nodes():
            if not n.name.startswith(rnn_scope_name):
                continue

            if n.name.endswith("sequence_length") and n.type == "Identity":
                log.debug("find non-const sequence length node")
            elif "CheckSeqLen" in n.name and n.is_const():
                # if seq length is const, the node might be const folded,
                # so we check this way.
                log.debug("find const sequence length node")
            else:
                continue
            seq_len_nodes.append(n)

        seq_len_node_cnt = len(seq_len_nodes)
        if seq_len_node_cnt == 0:
            return None
        if seq_len_node_cnt == 1:
            seq_len_node = seq_len_nodes[0]
            if seq_len_node.is_const():
                return seq_len_node
            # input of the "identity" node may be a "cast"
            # if so, then we have to keep it
            # sentence "math_ops.to_int32(sequence_length)" in tf results in the "cast" op
            if seq_len_node.inputs[0].type == "Cast":
                cast_node = seq_len_node.inputs[0]
                if not cast_node.inputs[0].name.startswith(rnn_scope_name):
                    return seq_len_node.inputs[0]
                raise ValueError("sequence length node should be outside of rnn scope")
            if not seq_len_node.inputs[0].name.startswith(rnn_scope_name):
                return seq_len_node.inputs[0]
            raise ValueError("sequence length node should be outside of rnn scope")
        raise ValueError("there are more sequence length nodes than expected")

    def get_rnn_input_blacklist(self, rnn_weights, rnn_props):
        var_init_nodes = []
        for _, init_input_id in rnn_props.var_initializers.items():
            init_node = self.g.get_node_by_output(init_input_id)
            var_init_nodes.append(init_node)

        # weight/bias inputs, and c/h initializers are dynamic_rnn/LSTMCell's parameters.
        # we will use them to filter out the dynamic_rnn's input tensor.
        blacklist_inputs = [rnn_weights.kernel.node, rnn_weights.bias.node, rnn_weights.forget_bias.node]
        blacklist_inputs.extend(var_init_nodes)

        return blacklist_inputs

    def find_inputs(self, rnn_scope_name, rnn_props, match, input_blacklist=None):
        rnn_input_nodes = []
        for n in self.g.get_nodes():
            if n.name.startswith(rnn_scope_name):
                # find input node that are not within rnn scope
                for input_id, input_node in zip(n.input, n.inputs):
                    if not input_node.name.startswith(rnn_scope_name):
                        if input_node not in input_blacklist:
                            rnn_input_nodes.append([input_node, input_id])

        if len(rnn_input_nodes) != 1:
            log.debug("found %d inputs for the dynamic_run, unexpected. They are %s",
                      len(rnn_input_nodes), rnn_input_nodes)
            return rnn_props

        input_node_candidate = rnn_input_nodes[0][0]
        input_id_candidate = rnn_input_nodes[0][1]

        # we should not limit the rnn_input_nodes' type be Placeholder or Const,
        # because there might some Reshape/etc. ops after the Placeholder
        rnn_props.input_node = input_node_candidate
        rnn_props.input_id = input_id_candidate
        return rnn_props

# process found info according to ONNX requirement
    def process_input_x(self, rnn_props, rnn_scope_name):
        self.print_step("look for possible transpose following RNN input node")
        # todo: peepholdes P is not considered now
        input_consumers = self.g.find_output_consumers(rnn_props.input_id)
        consumers_in_rnn_scope = []
        for consumer in input_consumers:
            if consumer.name.startswith(rnn_scope_name):
                consumers_in_rnn_scope.append(consumer)

        if len(consumers_in_rnn_scope) != 1:
            log.warning("RNN input node has %d onsumers in current rnn scope %s skip",
                        len(consumers_in_rnn_scope), rnn_scope_name)
            return None

        possible_transpose_after_input = consumers_in_rnn_scope[0]

        self.print_step("convert the transpose to onnx node if there is one found.")
        # check whether time_major is enabled or not
        # in TF, if time_major is not enabled, input format is [batch, time, ...]
        # but, during TF handling, at the beginning, the data will be transposed to [time, batch, ...]
        # after processing, the format is changed back before returning result.
        # So here, we judge the time_major by checking the transpose operator existence.
        converted_transpose = self._convert_timemajor_transpose(possible_transpose_after_input)
        if converted_transpose:
            log.debug("detect batch-major inputs")
            rnn_props.time_major = False
            rnn_props.x_input_id = converted_transpose.output[0]
            self.all_nodes.extend([converted_transpose])
        else:
            log.debug("detect timer-major inputs")
            rnn_props.time_major = True
            rnn_props.x_input_id = rnn_props.input_id

        rnn_props.onnx_input_ids["X"] = rnn_props.x_input_id
        return rnn_props

    def process_weights_and_bias(self, rnn_weights, rnn_props):
        pass

    def process_var_init_nodes(self, rnn_props):
        pass

    def process_seq_length(self, rnn_props, seq_length_node):
        # output: [time step, batch size, input size]
        shape_node = self.g.make_node("Shape", [rnn_props.x_input_id])

        # LSTMCell only allow inputs of [batch size, input_size], so we assume dynamic_rnn has 3 dims.
        # Slice cannot support Int64 in OPSET 7, so we cast here.
        cast_shape_node = self.g.make_node("Cast", [shape_node.output[0]],
                                           attr={"to": onnx_pb.TensorProto.FLOAT},
                                           shapes=[self.g.get_shape(shape_node.output[0])])

        batchsize_node = self.g.make_node("Slice", [cast_shape_node.output[0]],
                                          attr={"axes": [0], "starts": [1], "ends": [2]})

        # Tile's repeats must be INT64
        repeat_node = self.g.make_node("Cast", [batchsize_node.output[0]],
                                       attr={"to": onnx_pb.TensorProto.INT64})

        self.all_nodes.extend([shape_node, cast_shape_node, batchsize_node, repeat_node])

        if not seq_length_node:
            timestep_node = self.g.make_node("Slice", [cast_shape_node.output[0]],
                                             attr={"axes": [0], "starts": [0], "ends": [1]})

            tile_node = self.g.make_node("Tile", [timestep_node.output[0], repeat_node.output[0]])

            # LSTM sequence_lens needs to be int32
            seq_length_node = self.g.make_node('Cast', [tile_node.output[0]],
                                               attr={"to": onnx_pb.TensorProto.INT32})

            self.all_nodes.extend([timestep_node, tile_node, seq_length_node])

        rnn_props.onnx_input_ids["sequence_lens"] = seq_length_node.output[0]
        return seq_length_node, batchsize_node

    def process_outputs(self, match, rnn_node, rnn_props, rnn_scope_name):
        # There are 2 kinds of output nodes for dynamic_rnn
        # 1. output node, which ends with "Exit" followed
        #    either Transpose (when time_major is False),
        #    or TensorArrayGather
        # 2. cell_state node,
        #    2.1 if state_is_tuple is true:
        #        2.1.1 which ends with "Exit" followed by a Pack<C, H> whose name is out of rnn scope.
        #        2.1.2 which ends with "Exit" for c and h respectively, when cell_state.c/h is used.
        #    2.2 which ends with "Exit" if state_is_tuple is false
        for n in self.g.get_nodes():
            if n.type == "Exit" and n.name.startswith(rnn_scope_name):
                if len(n.input) != 1:
                    raise ValueError("exit's input count is " + str(len(n.input)) + " instead of 1")
                switch = n.inputs[0]
                if switch.type != "Switch":
                    log.debug("Exit has non-Switch input, skip.")
                    continue

                for var_name, funcs in self.switch_checkers.items():
                    var_checker = funcs[0]
                    var_exit_connector = funcs[1]

                    enter_target_input_id = self.check_switch_by_usage_pattern(switch, match, var_checker)
                    if enter_target_input_id:
                        log.debug("this is %s exit node", var_name)
                        var_exit_connector(rnn_node, n, rnn_props)
                        break

    def create_rnn_node(self, rnn_props):
        pass

# helper function
    def check_switch_by_usage_pattern(self, switch_node, match, check_func):
        if switch_node.type != 'Switch':
            return None

        # the first input is data
        merge_node = switch_node.inputs[0]
        if merge_node.type != "Merge":
            return None

        target_node_input_id = None
        for merge_input in merge_node.inputs:
            if merge_input.type == 'Enter':
                target_node_input_id = merge_input.input[0]
                log.debug("a Switch >> Merge >> Enter is found called %s", merge_input.inputs[0].name)
                break
            else:
                log.debug("skip the non-Enter input node of the merge_node")
                continue

        # check whether it is c_initialize or h_initialize
        if target_node_input_id:
            switch_consumers = self.g.find_output_consumers(switch_node.output[1])
            assert len(switch_consumers) == 1
            if switch_consumers[0].type == "Identity":
                identity_consumers = self.g.find_output_consumers(switch_consumers[0].output[0])
                return check_func(target_node_input_id, identity_consumers, match)
            log.error("not expected, skip ")
        log.warning("is_switch_used_by found no merge>>Enter node")

        return None

    @staticmethod
    def print_step(level_2, level_1="find_dynamic_run_unit"):
        log.debug(level_1 + " >> " + level_2)

    def _workaround_fill_ch_init_node(self, initializer_input_id, rnn_props):
        node = self.g.get_node_by_output(initializer_input_id)
        if node.type != "Fill":
            return None

        fill_val = node.inputs[1].get_tensor_value()[0]
        fill_val_dtype = utils.ONNX_TO_NUMPY_DTYPE[node.inputs[1].dtype]

        # this must be int64, since Concat's input data type must be consistent.
        num_direction_node = self.g.make_const(utils.make_name("Const"), np.array([1], dtype=np.float32))
        h_node = self.g.make_const(utils.make_name("Const"), np.array([rnn_props.hidden_size], dtype=np.float32))
        b_node = rnn_props.batch_size_node
        # Concat in OPSET7 does not support int64.
        tile_shape = self.g.make_node("Concat", [num_direction_node.output[0], b_node.output[0], h_node.output[0]],
                                      attr={"axis": 0})

        # Tile's repeats must be INT64
        attr = {"to": onnx_pb.TensorProto.INT64}
        tile_shape_int64 = self.g.make_node("Cast", [tile_shape.output[0]], attr)

        const_node = self.g.make_const(utils.make_name("Const"), np.array([[[fill_val]]], dtype=fill_val_dtype))
        tile_node = self.g.make_node("Tile", [const_node.output[0], tile_shape_int64.output[0]])
        self.all_nodes.extend([tile_shape, tile_shape_int64, tile_node])
        return tile_node

    def _convert_timemajor_transpose(self, node):
        if not check_is_timemajor_transpose(node):
            log.debug("not found timemajor transpose")
            return None

        log.debug("found timemajor transpose")

        attr = {"perm": np.array([1, 0, 2], dtype=np.int64)}
        new_trans = self.g.make_node("Transpose", [node.input[0]], attr=attr,
                                     shapes=[self.g.get_shape(node.output[0])],
                                     dtypes=[self.g.get_dtype(node.input[0])])
        self.g.replace_all_inputs(self.g.get_nodes(), node.output[0], new_trans.output[0])
        return new_trans
