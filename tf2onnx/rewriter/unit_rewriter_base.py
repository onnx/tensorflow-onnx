# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn_unit_base - lstm support
"""

from __future__ import division
from __future__ import print_function

from onnx import onnx_pb
from tf2onnx.rewriter.rnn_utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.rnn_unit_writer_base")


# dynamic_rnn or bidirectional_dynamic_rnn related logic will be mapped to this base class.
class UnitRewriterBase:
    def __init__(self, g):
        self.g = g
        self.all_nodes = self.g.get_nodes()
        # used to track nodes in rnn_scope_name to keep (e.g. not delete) for each single match run
        self.must_keep_nodes = []

        # checker signature : func_name(enter_target_node_input_id, identity_consumers, match)
        # exit connector signature: func_name(rnn_node, exit_node, rnn_props)
        self.switch_checkers = {}

    @staticmethod
    def print_step(level_2, level_1="find_dynamic_run_unit"):
        log.debug(level_1 + " >> " + level_2)

    def get_rnn_scope_name(self, match):
        pass

    def get_weight_and_bias(self, match):
        pass

    def process_input_x(self, rnn_props, rnn_scope_name):
        pass

    def process_weights_and_bias(self, rnn_weights, rnn_props):
        pass

    def process_var_init_nodes(self, rnn_props):
        pass

    def process_seq_length(self, rnn_props, seq_len_input_node):
        pass

    def create_rnn_node(self, rnn_props):
        pass

    def get_rnn_input_blacklist(self, rnn_weights, rnn_props):
        var_init_nodes = []
        for _, init_input_id in rnn_props.var_initializers.items():
            init_node = self.g.get_node_by_name(init_input_id)
            var_init_nodes.append(init_node)
            self.must_keep_nodes.append(init_node)

        # weight/bias inputs, and c/h initializers are dynamic_rnn/LSTMCell's parameters.
        # we will use them to filter out the dynamic_rnn's input tensor. 
        blacklist_inputs = [rnn_weights.kernel.node, rnn_weights.bias.node, rnn_weights.forget_bias.node]
        blacklist_inputs.extend(var_init_nodes)

        return blacklist_inputs

    def run(self, unit_type):
        # allow_reorder must be true. because LSTMCell and BasicLSTMCell's call function
        # are defining the calculation with different orders. Then we can share the same 
        # pattern.
        cell_pattern = get_pattern(unit_type)
        matcher = GraphMatcher(cell_pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(self.g.get_nodes()))

        if match_results:
            for match in match_results:
                self.run_single_match(match)
            self.print_step("finish handling")
            self.g.update_proto()
        return self.g.get_nodes()

    def run_single_match(self, match):
        log.debug("=========================")
        self.print_step("start handling a new potential rnn cell")
        self.all_nodes = self.g.get_nodes()
        self.must_keep_nodes = []

        rnn_scope_name = self.get_rnn_scope_name(match)
        if not rnn_scope_name:
            log.error("unable to find rnn scope name, skip")
            return REWRITER_RESULT.SKIP
        else:
            log.debug("rnn scope name is " + rnn_scope_name)

        self.print_step("get_weight_and_bias starts")
        rnn_weights = self.get_weight_and_bias(match)
        if not rnn_weights:
            log.error("rnn weights check failed, skip")
            return REWRITER_RESULT.SKIP

        rnn_props = RnnProperties()
        self.get_var_initializers(match, rnn_props, rnn_scope_name)
        if not rnn_props.var_initializers.keys:
            log.error("no cell variable initializers found, skip")
            return REWRITER_RESULT.SKIP

        seq_len_input_node = self.find_sequence_length_node(rnn_scope_name)
        input_filter = self.get_rnn_input_blacklist(rnn_weights, rnn_props)
        if seq_len_input_node:
            input_filter.append(seq_len_input_node)

        self.find_inputs(rnn_scope_name, rnn_props, input_filter)
        if not rnn_props.is_valid():
            log.error("rnn properties are not valid, skip")
            return REWRITER_RESULT.SKIP

        if not self.process_input_x(rnn_props, rnn_scope_name):
            log.error("rnn input x not found, skip")
            return REWRITER_RESULT.SKIP

        self.print_step("process the weights/bias/ft_bias, to fit onnx weights/bias requirements")
        self.process_weights_and_bias(rnn_weights, rnn_props)

        _, batch_size_node = self.process_seq_length(rnn_props, seq_len_input_node)
        rnn_props.batch_size_node = batch_size_node

        self.process_var_init_nodes(rnn_props)

        self.print_step("start to build new rnn node")

        rnn_node = self.create_rnn_node(rnn_props)
        self.all_nodes.append(rnn_node)

        self.print_step("start to handle outputs")
        self.process_outputs(match, rnn_node, rnn_props, rnn_scope_name)

        self.print_step("remove all nodes within original rnn scope except some nodes still useful")
        new_nodes = []
        for n in self.all_nodes:
            if n in self.must_keep_nodes:
                new_nodes.append(n)
                continue
            else:
                if n.name.startswith(rnn_scope_name):
                    pass
                else:
                    new_nodes.append(n)

        self.g.set_nodes(new_nodes)

    def find_inputs(self, rnn_scope_name, rnn_props, input_blacklist=None):
        rnn_input_nodes = []
        for n in self.g.get_nodes():
            if n.name.startswith(rnn_scope_name):
                # find input node that are not within rnn scope
                for input_id, input_node in zip(n.input, n.inputs):
                    if not input_node.name.startswith(rnn_scope_name):
                        if input_node not in input_blacklist:
                            rnn_input_nodes.append([input_node, input_id])

        if len(rnn_input_nodes) != 1:
            log.error("found " + str(len(rnn_input_nodes)) + " inputs for the dynamic_run, unexpected. They are ")
            log.error(rnn_input_nodes)
            return rnn_props

        input_node_candidate = rnn_input_nodes[0][0]
        input_id_candidate = rnn_input_nodes[0][1]

        # we should not limit the rnn_input_nodes' type be Placeholder or Const, 
        # because there might some Reshape/etc. ops after the Placeholder
        rnn_props.input_node = input_node_candidate
        rnn_props.input_id = input_id_candidate
        return rnn_props

    def get_var_initializers(self, match, rnn_props, rnn_scope_name):
        loop_cond_op = None
        for n in self.g.get_nodes():
            if n.type == 'LoopCond' and n.name.startswith(rnn_scope_name):
                if not loop_cond_op:
                    loop_cond_op = n
                else:
                    raise ValueError("only a LoopCond is expected to find in a dynamic run")

        if loop_cond_op is None:
            log.error("No LoopCond op is found, skip")
            return None

        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for n in switch_nodes:
            if n.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")

            for var_name, funcs in self.switch_checkers.items():
                var_checker = funcs[0]
                if funcs[2] == False:
                    continue
                enter_target_input_id = self.check_switch_by_usage_pattern(n, match, var_checker)
                if enter_target_input_id:
                    log.debug("found initializer node for " + var_name + ": " + enter_target_input_id)
                    rnn_props.var_initializers[var_name] = enter_target_input_id
                    break

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
                log.debug("a Switch >> Merge >> Enter is found called " + merge_input.inputs[0].name)
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
            else:
                log.error("not expected, skip ")
        else:
            log.warning("is_switch_used_by found no merge>>Enter node")

        return None

    def find_sequence_length_node(self, rnn_scope_name):
        # "sequence_length" under current rnn scope is the seq len node (if there is).
        # this is hardcoded in dynamic_rnn().

        seq_len_nodes = []
        for n in self.g.get_nodes():
            if not n.name.startswith(rnn_scope_name):
                continue

            if n.name.endswith("sequence_length") and n.type in ("Identity"):
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
            return
        elif seq_len_node_cnt == 1:
            seq_len_node = seq_len_nodes[0]
            if seq_len_node.is_const():
                self.must_keep_nodes.append(seq_len_node)
                return seq_len_node
            elif not seq_len_node.inputs[0].name.startswith(rnn_scope_name):
                return seq_len_node.inputs[0]
            else:
                raise ValueError("sequence length node should be outside of rnn scope")
        else:
            raise ValueError("there are more sequence length nodes than expected")

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
                        log.debug("this is " + var_name +" exit node")
                        var_exit_connector(rnn_node, n, rnn_props)
                        break




