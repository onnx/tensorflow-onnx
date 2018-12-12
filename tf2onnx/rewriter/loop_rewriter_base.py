# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter_base
"""

from __future__ import division
from __future__ import print_function
import copy
import logging
from collections import deque
from tf2onnx.rewriter.rnn_utils import is_loopcond_op, is_tensor_array_op, is_tensor_array_write_op
from tf2onnx.rewriter.rnn_utils import BodyGraphDict, REWRITER_RESULT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.loop_rewriter_base")

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class Context:
    def __init__(self):
        self.need_keep_nodes = []
        self.while_context_scope = None
        self.loop_variables = {}


class LoopVariable:
    def __init__(self, enter_name, enter_input_id, next_iteration_input_id,
                 switch_true_identity_output_id, exit_output_id, is_tensor_array):
        self.enter_name = enter_name
        self.enter_input_id = enter_input_id
        self.next_iteration_input_id = next_iteration_input_id
        self.switch_true_identity_output_id = switch_true_identity_output_id
        self.exit_output_id = exit_output_id

        self.is_tensor_array = is_tensor_array
        self.ta_index_id = None


class LoopRewriterBase:
    def __init__(self, g):
        self.g = g
        self.keep_nodes_global = []

    def create_context(self):
        return Context()

    def need_rewrite(self, context):
        return False

    def rewrite(self, context):
        return REWRITER_RESULT.FAIL

    def run_internal(self):
        log.debug("enter loop rewriter")
        for n in self.g.get_nodes():
            if is_loopcond_op(n):
                log.debug("======================")
                log.debug("found LoopCond op named %s", n.name)
                context = self.create_context()
                self._parse_loop_variables(n, context)
                if self.need_rewrite(context):
                    _result = self.rewrite(context)
                    if _result == REWRITER_RESULT.OK:
                        log.debug("rewrite successfully")
                    elif _result == REWRITER_RESULT.SKIP:
                        log.debug("rewrite skipped for LoopCond called %s", n.name)
                        continue
                    elif _result == REWRITER_RESULT.FAIL:
                        raise ValueError("rewrite failed, so just fast fail it")
        all_output_name = copy.deepcopy(self.g.output_names)
        if all_output_name:
            all_output_name.extend(BodyGraphDict.get_body_graph_output_names())
            self.g.delete_unused_nodes(all_output_name)
        return self.g.get_nodes()

    def _parse_loop_variables(self, loop_cond_op, context):
        parts = loop_cond_op.name.split('/')
        context.while_context_scope = '/'.join(parts[0:-1]) + "/"
        log.debug("found while loop scope %s", context.while_context_scope)

        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for s in switch_nodes:
            if s.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")

            loop_var = self._get_loop_var_from_switch(s)
            if loop_var.enter_name in context.loop_variables:
                raise ValueError("duplicated enter name registered")

            context.loop_variables[loop_var.enter_name] = loop_var

    def _get_loop_var_from_switch(self, switch_node):
        if switch_node.type != 'Switch':
            log.error("not a switch node, skip")
            return None

        # the first input is data
        merge_node = switch_node.inputs[0]
        if merge_node.type != "Merge":
            log.error("switch node does not has Merge as its first input")
            return None

        # find the output_true consumers
        switch_consumers = self.g.find_output_consumers(switch_node.output[1])
        if len(switch_consumers) != 1:
            raise ValueError("switch has non-1 consumers")

        if switch_consumers[0].type != "Identity":
            raise ValueError("switch has consumer that is not Identity")
        identity_node = switch_consumers[0]

        target_node_input_id = None
        enter_node = [n for n in merge_node.inputs if n.type == 'Enter'][0]
        target_node_input_id = enter_node.input[0]
        log.debug("a Switch >> Merge >> Enter is found called %s", enter_node.inputs[0].name)

        next_iteration_node = [n for n in merge_node.inputs if n.type == 'NextIteration'][0]
        last_iteration_output_id = next_iteration_node.input[0]

        # find the output_false consumers to see whether there is consumer for this var
        switch_false_consumers = self.g.find_output_consumers(switch_node.output[0])
        false_consumer_count = len(switch_false_consumers)
        exit_output_id = None
        if false_consumer_count == 1:
            exit_node = switch_false_consumers[0]
            if exit_node.type != "Exit":
                raise ValueError("switch false branch is followed by non-Exit")
            exit_output_id = exit_node.output[0]
        elif false_consumer_count == 0:
            exit_output_id = None
        else:
            raise ValueError("unexpected number of switch false consumers")

        is_ta = False
        if is_tensor_array_op(self.g.get_node_by_output(target_node_input_id)):
            is_ta = True

        loop_var = LoopVariable(enter_node.name, target_node_input_id, last_iteration_output_id,
                                identity_node.output[0], exit_output_id, is_ta)
        loop_var = self._tune_shape_for_loop_var(loop_var)
        loop_var = self._tune_shape_for_loop_ta_var(loop_var)
        return loop_var

    def _tune_shape_for_loop_ta_var(self, loop_var):
        if loop_var.is_tensor_array:
            ta_write_node = self.g.get_node_by_output(loop_var.next_iteration_input_id)
            if not is_tensor_array_write_op(ta_write_node):
                raise ValueError("ta var nextiteration is not following ta write op")

            loop_var.next_iteration_input_id = ta_write_node.input[2]
            loop_var.ta_index_id = ta_write_node.input[1]

            ta_output_shape = None
            next_iteration_shape = self.g.get_shape(loop_var.next_iteration_input_id)
            if next_iteration_shape is None:
                enter_node = ta_write_node.inputs[0]
                ta_node_output = enter_node.input[0]
                ta_element_shape = self.g.get_shape(ta_node_output)
                ta_output_shape = ta_element_shape
                log.debug("loop var [%s, %s] output shapes are inferred from TA element shape", loop_var.enter_name,
                          loop_var.enter_input_id)
            else:
                log.debug("loop var [%s, %s] output shapes are inferred from cell output %s", loop_var.enter_name,
                          loop_var.enter_input_id, loop_var.next_iteration_input_id)
                ta_output_shape = next_iteration_shape

            self.g.set_shape(loop_var.next_iteration_input_id, ta_output_shape)
            self.g.set_shape(loop_var.switch_true_identity_output_id, ta_output_shape)
            self.g.set_shape(loop_var.exit_output_id, ta_output_shape)

        return loop_var

    def _tune_shape_for_loop_var(self, loop_var):
        if loop_var.is_tensor_array:
            return loop_var
        log.debug("_tune_shape_for_loop_var for loop var [%s, %s, %s]", loop_var.enter_name,
                  loop_var.enter_input_id, loop_var.next_iteration_input_id)
        var_output_shape = self.g.get_shape(loop_var.enter_input_id)
        if var_output_shape is None:
            var_output_shape = self.g.get_shape(loop_var.next_iteration_input_id)

        self.g.set_shape(loop_var.next_iteration_input_id, var_output_shape)
        self.g.set_shape(loop_var.switch_true_identity_output_id, var_output_shape)
        self.g.set_shape(loop_var.exit_output_id, var_output_shape)
        log.debug("_tune_shape_for_loop_var new shape is %s", var_output_shape)

        return loop_var

    @staticmethod
    def find_subgraph(graph_meta, g):
        input_ids = graph_meta.input_ids
        if graph_meta.other_enter_input_ids:
            input_ids += graph_meta.other_enter_input_ids

        input_ids = set(input_ids)
        output_ids = set(graph_meta.output_ids)
        log.debug("input ids %s ", input_ids)
        log.debug("output ids %s ", output_ids)
        nodes = []
        q = deque()
        output_nodes = [g.get_node_by_output(output_id) for output_id in output_ids]
        handled_nodes = []
        q.extend(output_nodes)
        nodes.extend(output_nodes)
        enter_nodes = set()
        while q:
            n = q.popleft()
            if not n:
                continue
            if n in handled_nodes:
                continue

            handled_nodes.append(n)

            n_inputs = set(n.input)
            for i in n_inputs:
                input_node = g.get_node_by_output(i)
                if i in input_ids:
                    log.debug("terminate the input search at %s", i)
                elif not input_node:
                    if i in g.model_inputs:
                        log.debug("find a model input, which might be a placeholder")
                    elif g.is_initializer(i):
                        log.debug("find an initializer, this might be generated during op conversion")
                    else:
                        log.error("input node does not exist, node name is: [%s] ", i)
                        raise ValueError("failed to get input")
                elif input_node.type == "Enter":
                    enter_nodes.add(input_node)
                    log.debug("terminate the input search at %s", i)
                else:
                    log.debug("add node %s into sub graph %s", input_node.name, n.name)
                    if input_node not in nodes:
                        nodes.append(input_node)
                    q.append(input_node)

            implicit_inputs = n.get_implicit_inputs(require_input_in_cur_graph=True)
            n_inputs = set(implicit_inputs)
            for i in n_inputs:
                input_node = g.get_node_by_output(i)
                if i in input_ids:
                    log.debug("terminate the input search at %s", i)
                elif not input_node:
                    log.debug("implicit input is initializer or input in main graph, node name is: [%s] ", i)
                elif input_node.type == "Enter":
                    enter_nodes.add(input_node)
                    log.debug("terminate the input search at %s", i)
                else:
                    log.debug("add node %s into sub graph %s", input_node.name, n.name)
                    if input_node not in nodes:
                        nodes.append(input_node)
                    q.append(input_node)

        return nodes, enter_nodes
