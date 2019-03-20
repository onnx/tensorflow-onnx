# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.unit_rnn_rewriter_base
"""

from __future__ import division
from __future__ import print_function
import logging
from onnx import TensorProto
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase, Context
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT, get_pattern, \
    get_rnn_scope_name, parse_rnn_loop, is_select_op, is_tensor_array_write_op, \
    seq_len_pattern
from tf2onnx.graph_matcher import GraphMatcher


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.unit_rnn_rewriter_base")


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access

class UnitRnnContext(Context):
    def __init__(self):
        super(UnitRnnContext, self).__init__()
        self.rnn_scope = None
        self.cell_match = None  # matched cell

        self.weights = None
        self.seq_len_node = None
        self.state_variables = {}
        self.input_size = None
        self.hidden_size = None

        self.attributes = {}
        self.onnx_input_ids = {}  # onnx inputs: [X, W, R, B, sequence_lens, initial_h, initial_c, P]
        self.activations = None


class UnitRnnRewriterBase(LoopRewriterBase):
    """
    main procedures:
    1 extract info of while_loop based on loop_rewriter_base
    2 check whether extracted loop is a unit rnn, fall back in necessity:
        1 parse rnn scope name
        2 check if it's a dynamic_rnn
        3 find needed info from tensorflow graph
    3 process found info according to ONNX requirement
    """
    def __init__(self, g):
        super(UnitRnnRewriterBase, self).__init__(g)
        # {var_name: (finder, connector)}
        self.state_variable_handler = None
        self.state_variable_handlers = None

    def create_context(self):
        return UnitRnnContext()

    def run(self):
        log.debug("enter unit rnn rewriter base")
        return self.run_internal()

    def need_rewrite(self, context):
        context.rnn_scope = get_rnn_scope_name(context.while_context_scope)

        if not parse_rnn_loop(self.g, context.loop_properties, context.rnn_scope,
                              context.while_context_scope):
            log.debug("parse_rnn_loop failed, SKIP")
            return False

        if not self.parse_unit_rnn(context):
            log.debug("failed to parse unit rnn, SKIP")
            return False

        if not self.is_valid(context):
            log.debug("parsed rnn is not valid, SKIP")
            return False
        return True

    def is_valid(self, context):
        return True

    def parse_unit_rnn(self, context):
        """
        parse needed info from tensorflow graph:
        1 weight
        2 state variables used in rnn unit, such as c_t, h_t
        3 sequence node
        4 input_x
        5 activations, optional
        6 attributes, optional
        """
        log.debug("parse unit rnn")

        log.debug("match unit cell against loop body graph")
        cell_match = self.find_cell(context)
        if not cell_match:
            log.debug('failed to match cell pattern')
            return False
        context.cell_match = cell_match

        log.debug("get_weight_and_bias starts")
        rnn_weights = self.get_weight_and_bias(context)
        if not rnn_weights:
            log.debug("rnn weights check failed, SKIP")
            return False
        context.weights = rnn_weights

        if not self.get_state_variables(context):
            log.debug("no cell variable initializers found, SKIP")
            return False

        seq_len_node = self.find_sequence_length_node(context)
        if seq_len_node:
            log.debug("find sequence node: %s", seq_len_node.name)
            context.seq_len_node = seq_len_node

        # require exact one input
        inputs = context.loop_properties.scan_inputs_initial_values
        if len(inputs) != 1:
            log.debug("found %d inputs for the unit rnn: %s",
                      len(inputs), inputs)
            return False
        context.onnx_input_ids["X"] = inputs[0]

        if not self.get_rnn_activation(context):
            log.debug("failed to find activation")
            return False

        if not self.get_attributes(context):
            log.debug("wrong attributes found")
            return False

        return True

    def find_cell(self, context):
        raise NotImplementedError()

    def _match_cell(self, context, unittype):
        """match unit cell"""
        cell_pattern = get_pattern(unittype)
        matcher = GraphMatcher(cell_pattern, allow_reorder=True)

        loop_props = context.loop_properties
        inputs = loop_props.state_inputs + loop_props.scan_inputs
        input_ids = [input_tensor_value_info.id for input_tensor_value_info in inputs]
        outputs = loop_props.state_outputs + loop_props.scan_outputs
        output_ids = [out_tensor_value_info.id for out_tensor_value_info in outputs]
        body_graph_ops, _, _ = LoopRewriterBase.find_subgraph(
            set(input_ids),
            set(output_ids),
            self.g, merge_as_end=True
        )

        match_results = list(matcher.match_ops(body_graph_ops))
        if len(match_results) != 1:
            return None
        return match_results[0]

    def get_weight_and_bias(self, context):
        raise NotImplementedError()

    def get_rnn_activation(self, context):
        return True

    def get_attributes(self, context):
        return True

    def rewrite(self, context):
        log.debug("enter unit rnn rewrite function")

        log.debug("process the weights/bias/ft_bias, to fit onnx weights/bias requirements")
        self.process_weights_and_bias(context)

        self.process_seq_length(context)

        self.process_var_init_nodes(context)

        log.debug("start to build new rnn node")

        rnn_node = self.create_rnn_node(context)
        context.rnn_node = rnn_node

        log.debug("start to handle outputs")
        # format of ONNX output is different with tf
        self.process_outputs(context)

        log.debug("rewrite successfully")
        return REWRITER_RESULT.OK

    def get_state_variables(self, context):
        """
        Get state variables by provided handlers. There maybe several handlers corresponding to
        different patterns of state variables.
        The commone method is to find state variables from loop property according to its
        next_iteration_input and switch_true_identity_output, see lstm_rewriter_v2
        """
        for handler in self.state_variable_handlers:
            can_handle = True
            for var_name, funcs in handler.items():
                finder = funcs[0]
                state_variable = finder(context)
                if state_variable:
                    log.debug("found state variable %s", var_name)
                    context.state_variables[var_name] = state_variable
                else:
                    log.debug("failed to get state variable %s", var_name)
                    can_handle = False
                    break
            if can_handle:
                self.state_variable_handler = handler
                return True
        return False

    def find_sequence_length_node(self, context):
        # get any state variable
        state_variable = list(context.state_variables.values())[0]
        next_iter_input_node = self.g.get_node_by_output(state_variable.next_iteration_input.id)
        if not is_select_op(next_iter_input_node):
            log.debug("no sequence length node is given")
            return None
        matcher = GraphMatcher(seq_len_pattern)
        match_result = matcher.match_op(next_iter_input_node)
        if not match_result:
            raise RuntimeError("failed to find sequence length.")
        return match_result.get_op("seq_len_node")

    def process_weights_and_bias(self, context):
        raise NotImplementedError()

    def process_seq_length(self, context):
        # output: [time step, batch size, input size]
        seq_len_node = context.seq_len_node
        shape_node = self.g.make_node("Shape", [context.onnx_input_ids["X"]])
        # LSTMCell only allow inputs of [batch size, input_size], so we assume dynamic_rnn has 3 dims.
        # Slice cannot support Int64 in OPSET 7, so we cast here.
        cast_shape_node = self.g.make_node(
            "Cast", [shape_node.output[0]],
            attr={"to": TensorProto.FLOAT},
            shapes=[self.g.get_shape(shape_node.output[0])]
        )
        batchsize_node = self.g.make_node(
            "Slice", [cast_shape_node.output[0]],
            attr={"axes": [0], "starts": [1], "ends": [2]}
        )
        if not seq_len_node:
            # Tile's repeats must be INT64
            repeat_node = self.g.make_node(
                "Cast", [batchsize_node.output[0]],
                attr={"to": TensorProto.INT64}
            )
            timestep_node = self.g.make_node(
                "Slice", [cast_shape_node.output[0]],
                attr={"axes": [0], "starts": [0], "ends": [1]}
            )
            tile_node = self.g.make_node("Tile", [timestep_node.output[0], repeat_node.output[0]])

            # LSTM sequence_lens needs to be int32
            seq_len_node = self.g.make_node(
                "Cast", [tile_node.output[0]],
                attr={"to": TensorProto.INT32}
            )
        context.onnx_input_ids["sequence_lens"] = seq_len_node.output[0]

    def process_var_init_nodes(self, context):
        raise NotImplementedError()

    def create_rnn_node(self, context):
        raise NotImplementedError()

    def process_outputs(self, context):
        for var_name, funcs in self.state_variable_handler.items():
            output_connector = funcs[1]
            output_connector(context)
            log.debug("connect output of %s to graph", var_name)

        self.connect_unit_rnn_output_to_graph(context)

    def connect_unit_rnn_output_to_graph(self, context):
        outputs = context.loop_properties.scan_outputs_exits
        if not outputs:
            log.debug("no one consume output")
            return

        gather_output_id = outputs[0].id
        log.debug("found output for lstm: %s", gather_output_id)

        # in tf batch major mode, output shape is : [batch, time, hidden]
        # in time major mode, output shape is: [time, batch, hidden]
        # in onnx, output shape is : [time, num_directions, batch, hidden]

        lstm_node = context.rnn_node
        output_id = lstm_node.output[0]
        lstm_output_shape = self.g.get_shape(output_id)
        squeeze_output_shape = [lstm_output_shape[0], lstm_output_shape[2], lstm_output_shape[3]]
        squeeze_node = self.g.make_node("Squeeze", [output_id], attr={"axes": [1]},
                                        shapes=[squeeze_output_shape],
                                        dtypes=[self.g.get_dtype(output_id)])
        self.g.replace_all_inputs(self.g.get_nodes(), gather_output_id, squeeze_node.output[0])

    def _find_state_variable_with_select(self, context,
                                         next_iteration_input,
                                         switch_true_identity_consumers):
        """
        Find state variables from switch_true_identity_consumers to next_iteration_input.
        Select maybe added after next_iteration_input.
        """
        # find all select not followed by TensorArrayWrite
        select = []
        for c in self.g.find_output_consumers(next_iteration_input):
            if not is_select_op(c):
                continue
            out_ta_writer = [
                o for o in self.g.find_output_consumers(c.output[0]) if is_tensor_array_write_op(o)
            ]
            if out_ta_writer:
                continue
            select.append(c)
        if len(select) == 1:
            next_iteration_input = select[0].output[0]
            switch_true_identity_consumers.append(select[0])

        log.debug(
            "try to find state variable from [%s, %s]",
            next_iteration_input,
            switch_true_identity_consumers
        )

        def checker(state_variable):
            if state_variable.next_iteration_input.id != next_iteration_input:
                return False
            for consumer in switch_true_identity_consumers:
                if state_variable.switch_true_identity_output.id not in consumer.input:
                    return False
            return True

        state_variables = context.loop_properties.get_variables(checker)
        if len(state_variables) != 1:
            log.debug("found %d state variables", len(state_variables))
            return None
        return state_variables[0]
