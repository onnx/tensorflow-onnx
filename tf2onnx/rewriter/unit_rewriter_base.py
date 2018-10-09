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

    @staticmethod
    def print_step(level_2, level_1="find_dynamic_run_unit"):
        log.info(level_1 + " >> " + level_2)

    def ct_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        pass

    def ht_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        pass

    # when state is not tuple, ct and ht may share same switch.
    def ct_ht_shared_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        pass

    def get_rnn_scope_name(self, match):
        pass

    def get_weight_and_bias(self, match):
        pass

    def process_weights_and_bias(self, rnn_weights):
        pass

    def process_output_connectors(self, match, lstm_node, rnn_props, rnn_scope_name):
        pass

    def get_rnn_input_blacklist(self, rnn_weights, rnn_inits):
        if rnn_inits.share_init_node:
            ch_node = self.g.get_node_by_name(rnn_inits.share_init_input_id)
            c_h_nodes = [ch_node]
            self.must_keep_nodes.append(ch_node)
        else:
            c_h_nodes = [self.g.get_node_by_name(rnn_inits.c_init_input_id), 
                         self.g.get_node_by_name(rnn_inits.h_init_input_id)]

        # weight/bias inputs, and c/h initializers are dynamic_rnn/LSTMCell's parameters.
        # we will use them to filter out the dynamic_rnn's input tensor. 
        blacklist_inputs = [rnn_weights.kernel.node, rnn_weights.bias.node, rnn_weights.forget_bias.node]
        blacklist_inputs.extend(c_h_nodes)

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
        log.info("=========================")
        self.print_step("start handling a new potential LSTM Cell")
        self.all_nodes = self.g.get_nodes()
        self.must_keep_nodes = []

        rnn_scope_name = self.get_rnn_scope_name(match)
        if not rnn_scope_name:
            log.error("unable to find rnn scope name, skip")
            return REWRITER_RESULT.SKIP
        else:
            log.info("rnn scope name is " + rnn_scope_name)

        self.print_step("get_weight_and_bias starts")
        rnn_weights = self.get_weight_and_bias(match)
        if not rnn_weights:
            log.error("basic LSTM Cell weights check failed, skip")
            return REWRITER_RESULT.SKIP

        rnn_inits = self.get_ct_ht_initializers(match, rnn_scope_name)
        if not rnn_inits:
            log.error("basic LSTM Cell ct/ht initializer check failed, skip")
            return REWRITER_RESULT.SKIP

        input_filter = self.get_rnn_input_blacklist(rnn_weights, rnn_inits)

        rnn_props = self.find_input_and_connectors(rnn_scope_name, input_filter)
        if not rnn_props.is_valid():
            log.error("RNN properties are not valid, skip")
            return REWRITER_RESULT.SKIP

        if not self.process_lstm_input_x(rnn_props, rnn_scope_name):
            log.error("RNN input x not found, skip")
            return REWRITER_RESULT.SKIP

        self.print_step("process the weights/bias/ft_bias, to fit onnx weights/bias requirements")
        W, R, B, input_size, hidden_size = self.process_weights_and_bias(rnn_weights)
        rnn_props.input_size = input_size
        rnn_props.hidden_size = hidden_size

        # create node
        w_name = utils.make_name("W")
        w_node = self.g.make_const(w_name, W, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = self.g.make_const(r_name, R, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = self.g.make_const(b_name, B, skip_conversion=True)

        init_h_id = None
        init_c_id = None
        if rnn_inits.share_init_node:
            init_h_id, init_c_id = self.process_non_tuple_ch_init_nodes(rnn_inits.share_init_input_id, hidden_size)
        else:
            init_h_id, init_c_id = self.process_tuple_ch_init_nodes(rnn_inits.c_init_input_id,
                                                                    rnn_inits.h_init_input_id, hidden_size)
        assert init_h_id and init_c_id

        len_node = self.create_seq_len_node(rnn_props)

        self.print_step("start to build new LSTM node")

        # specify if the RNN is forward, reverse, or bidirectional.
        # Must be one of forward (default), reverse, or bidirectional.
        # Here we won't mark bidirectional, we will have another rewriter running after this one, which will based 
        # on patterns to combine a forward LSTM and a backward LSTM into a bidirectional one.
        direction = "forward"
        if rnn_props.is_backward:
            direction = "reverse"
        # todo: input_forget
        attr = {"direction": direction, "hidden_size": hidden_size}
        lstm_input_nodes = [rnn_props.x_node, w_node, r_node, b_node, len_node]
        lstm_inputs = list(map(lambda n: n.output[0], lstm_input_nodes))
        lstm_inputs.extend([init_h_id, init_c_id])
        lstm_node = make_onnx_node(self.g, "LSTM", lstm_inputs, attr, 3)
        self.all_nodes.extend([lstm_node])

        self.print_step("start to handle output connectors")
        self.process_output_connectors(match, lstm_node, rnn_props, rnn_scope_name)

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

    def find_input_and_connectors(self, rnn_scope_name, input_blacklist=None):
        rnn_props = RnnProperties()
        rnn_input_nodes = []
        connector_nodes = []
        for n in self.g.get_nodes():
            if n.name.startswith(rnn_scope_name):
                # find input node that are not within rnn scope
                for input_id, input_node in zip(n.input, n.inputs):
                    if not input_node.name.startswith(rnn_scope_name):
                        if input_node not in input_blacklist:
                            rnn_input_nodes.append([input_node, input_id])

                # find output consumers that are not within rnn scope
                for output_name in n.output:
                    output_nodes = self.g.find_output_consumers(output_name)
                    for out_node in output_nodes:
                        if not out_node.name.startswith(rnn_scope_name):
                            connector_nodes.append(out_node)

        if len(rnn_input_nodes) != 1:
            log.error("found 2 inputs for the dynamic_run, unexpected. They are ")
            log.error(rnn_input_nodes)
            return rnn_props

        input_node_candidate = rnn_input_nodes[0][0]
        input_id_candidate = rnn_input_nodes[0][1]

        # in TF bidirectional_rnn, backforward, will first reverse the inputs, then dynamic_run, then reverse 
        # output back. And the 2 reverses operators are not within the deeper dynamic_rnn scope. 
        # So, in this case, we might get a ReverseV2 op in rnn_input_nodes. 
        if input_node_candidate.type in ["ReverseV2"]:
            log.info("found reverse pattern")
            rnn_props.is_backward = True

            # ReverseV2 has 2 inputs, the second is axis.
            rnn_props.input_node = input_node_candidate.inputs[0]
            rnn_props.input_id = input_node_candidate.input[0]
            rnn_props.connectors = connector_nodes
            return rnn_props
        else:
            # we should not limit the rnn_input_nodes' type be PlaceHolder or Const, 
            # because there might some Reshape/etc. ops after the Placeholder
            rnn_props.input_node = input_node_candidate
            rnn_props.input_id = input_id_candidate
            rnn_props.connectors = connector_nodes
            return rnn_props

    def get_ct_ht_initializers(self, match, rnn_scope_name):
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

        # be noted: dynamic_rnn's initial_state can be constant or not.
        h_initializer = None
        c_initializer = None
        shared_ch_initializer_input_id = None  # only for non-tuple c_h initializer
        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for n in switch_nodes:
            if n.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")
            enter_target_input_id = self.check_switch_by_usage_pattern(n, match, self.ct_switch_check)
            if enter_target_input_id:
                c_initializer = enter_target_input_id
                continue

            enter_target_input_id = self.check_switch_by_usage_pattern(n, match, self.ht_switch_check)
            if enter_target_input_id:
                h_initializer = enter_target_input_id
                continue

            enter_target_input_id = self.check_switch_by_usage_pattern(n, match, self.ct_ht_shared_switch_check)
            if enter_target_input_id:
                shared_ch_initializer_input_id = enter_target_input_id
                continue

        # when shared_ch_initializer_input_id is not None, c_initializer and h_initializer
        # should be None, and vice versa
        if shared_ch_initializer_input_id:
            assert not c_initializer and not h_initializer
        else:
            assert not shared_ch_initializer_input_id

        return RnnInitializers(c_initializer, h_initializer, shared_ch_initializer_input_id)

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

    def process_lstm_input_x(self, rnn_props, rnn_scope_name):
        self.print_step("look for possible transpose following RNN input node")
        # todo: peepholdes P is not considered now
        input_consumers = self.g.find_output_consumers(rnn_props.input_id)
        cur_rnn_consumers = []
        for consumer in input_consumers:
            if not rnn_props.is_backward:
                if consumer.name.startswith(rnn_scope_name):
                    cur_rnn_consumers.append(consumer)
            else:
                # reversev2 might have a different name scope, so we check this way.
                if consumer.type == "ReverseV2":
                    cur_rnn_consumers.append(consumer)

        if len(cur_rnn_consumers) != 1:
            log.error("RNN input node has " + str(len(cur_rnn_consumers)) +
                      " consumers in current rnn scope " + rnn_scope_name + ", skip")
            return None

        possible_transpose_after_input = cur_rnn_consumers[0]
        if rnn_props.is_backward:
            assert possible_transpose_after_input.type == "ReverseV2"
            self.must_keep_nodes.append(possible_transpose_after_input)
            reverse_outputs = self.g.find_output_consumers(possible_transpose_after_input.output[0])
            assert len(reverse_outputs) == 1  # bidirectional_dynamic_rnn logic will promise this
            possible_transpose_after_input = reverse_outputs[0]

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
            rnn_props.x_node = converted_transpose
            self.all_nodes.extend([converted_transpose])
        else:
            log.debug("detect timer-major inputs")
            rnn_props.time_major = True
            rnn_props.x_node = rnn_props.input_node

        return rnn_props

    def _convert_timemajor_transpose(self, node):
        if not check_is_timemajor_transpose(node):
            log.debug("not found timemajor transpose")
            return

        attr = {"perm": np.array([1, 0, 2], dtype=np.int64)}
        new_trans = make_onnx_node(self.g, "Transpose", [node.input[0]], attr)

        self.g.copy_shape(node.output[0], new_trans.output[0])
        self.g.replace_all_inputs(self.g.get_nodes(), node.output[0], new_trans.output[0])
        return new_trans

    # todo: refine when implementing GRU
    def process_non_tuple_ch_init_nodes(self, input_id, hidden_size):

        attr = {"axes": [1], "starts": [0], "ends": [hidden_size]}
        slice_node1 = make_onnx_node(self.g, "Slice", [input_id], attr)

        unsqueeze_node_1 = make_onnx_node(self.g, "Unsqueeze", [slice_node1.output[0]], attr={"axes": [0]})

        attr = {"axes": [1], "starts": [hidden_size], "ends": [hidden_size*2]}
        slice_node2 = make_onnx_node(self.g, "Slice", [input_id], attr)

        unsqueeze_node_2 = make_onnx_node(self.g, "Unsqueeze", [slice_node2.output[0]], attr={"axes": [0]})

        self.all_nodes.extend([slice_node1, slice_node2, unsqueeze_node_1, unsqueeze_node_2])
        return unsqueeze_node_1.output[0], unsqueeze_node_2.output[0]

    # todo: refine when implementing GRU
    def process_tuple_ch_init_nodes(self, c_init_input_id, h_init_input_id, hidden_size):
        h_node_output = self.connect_initializer_node(h_init_input_id, hidden_size)
        c_node_output = self.connect_initializer_node(c_init_input_id, hidden_size)
        return h_node_output, c_node_output

    def connect_initializer_node(self, initializer_input_id, hidden_size):
        node = self.g.get_node_by_name(initializer_input_id)
        if node.is_const():
            val = node.get_tensor_value()
            initial_name = utils.make_name("Const")
            new_val = np.expand_dims(val, axis=0)
            const_node = self.g.make_const(initial_name, new_val)
            return const_node.output[0]
        else:
            squeeze_node = make_onnx_node(self.g, "Unsqueeze", [initializer_input_id], attr={"axes": [0]})
            self.g.replace_all_inputs(self.g.get_nodes(), initializer_input_id, squeeze_node.output[0])
            self.all_nodes.append(squeeze_node)
            return squeeze_node.output[0]

    def create_seq_len_node(self, rnn_props):
        # check whether input_node has valid shape
        if rnn_props.input_node.shape:
            self.print_step("input node has shape: parse batch_size, time_step, input_size, create seq_len const nodes")
            if rnn_props.time_major and rnn_props.input_node.shape[1] != utils.ONNX_UNKNOWN_DIMENSION:
                time_step = rnn_props.input_node.shape[0]
                batch_size = rnn_props.input_node.shape[1]
                input_size_2 = rnn_props.input_node.shape[2]
            elif rnn_props.input_node.shape[0] != utils.ONNX_UNKNOWN_DIMENSION:
                batch_size = rnn_props.input_node.shape[0]
                time_step = rnn_props.input_node.shape[1]
                input_size_2 = rnn_props.input_node.shape[2]

            assert rnn_props.input_size == input_size_2
            # todo: what if batch_size = -1
            sequence_lens = np.array([time_step for l in np.arange(batch_size)], dtype=np.int32)
            len_name = utils.make_name("Const")
            len_node = self.g.make_const(len_name, sequence_lens, skip_conversion=True)
            self.g.set_shape(len_node.output[0], sequence_lens.shape)
        else:
            self.print_step("prepare input nodes for new lstm node")
            shape_node = make_onnx_node(self.g, "Shape", [rnn_props.input_id])

            # LSTMCell only allow inputs of [batch, input_size], so we assume dynamic_rnn has 3 dims.
            # Slice cannot support Int64 in OPSET 7, so we cast here.
            attr = {"to": onnx_pb.TensorProto.FLOAT}
            cast_shape_node = make_onnx_node(self.g, "Cast", [shape_node.output[0]], attr)
            self.g.copy_shape(shape_node.output[0], cast_shape_node.output[0])

            attr = {"axes": [0], "starts": [0], "ends": [1]}
            batchsize_node = make_onnx_node(self.g, "Slice", [cast_shape_node.output[0]], attr)

            attr = {"to": onnx_pb.TensorProto.INT64}
            repeat_node = make_onnx_node(self.g, 'Cast', [batchsize_node.output[0]], attr)

            attr = {"axes": [0], "starts": [1], "ends": [2]}
            timestep_node = make_onnx_node(self.g, 'Slice', [cast_shape_node.output[0]], attr)

            tile_node = make_onnx_node(self.g, 'Tile', [timestep_node.output[0], repeat_node.output[0]])

            attr = {"to": onnx_pb.TensorProto.INT32}  # LSTM sequence_lens needs to be int32
            cast_back_node = make_onnx_node(self.g, 'Cast', [tile_node.output[0]], attr)

            len_node = cast_back_node
            self.all_nodes.extend([shape_node, cast_shape_node,
                                   timestep_node, batchsize_node, repeat_node, tile_node, len_node])

        return len_node
