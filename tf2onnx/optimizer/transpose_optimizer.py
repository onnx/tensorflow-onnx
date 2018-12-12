# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
"""Transpose Optimizer."""

from __future__ import unicode_literals

import logging

import numpy as np
from onnx import numpy_helper

from tf2onnx import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.optimizer.transpose_optimizer")


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring
# FIXME:
# pylint: disable=unused-variable

def is_nhwc_transpose(transpose_node):
    perm_attr = transpose_node.get_attr('perm')
    return transpose_node.type == "Transpose" and perm_attr and perm_attr.ints == [0, 2, 3, 1]


def is_nchw_transpose(transpose_node):
    perm_attr = transpose_node.get_attr('perm')
    return transpose_node.type == "Transpose" and perm_attr and perm_attr.ints == [0, 3, 1, 2]


def is_useless_transpose(transpose_node):
    perm_attr = transpose_node.get_attr('perm')
    return transpose_node.type == "Transpose" and perm_attr and perm_attr.ints == list(range(len(perm_attr.ints)))


class TransposeOptimizer(object):
    """Transpose Optimizer."""

    def __init__(self, graph, output_names, debug=False):
        self._g = graph
        self._output_names = [name.split(":")[0] for name in output_names]
        self._debug = debug
        self._handler_map = {}
        self._force_stop = {}

        # make sure all proto of nodes or attribtues are update to date
        self._g.update_proto()
        self._initialize_handlers()
        self.pre_optimize_action()

    @property
    def nodes(self):
        return self._g.get_nodes()

    def pre_optimize_action(self):
        # make Reshape into a const, which then can be fused into Conv's weight for mobilenet_v1_75_192
        ops = self.nodes
        constable_reshape_ops = [n for n in ops
                                 if (n.type == "Reshape"
                                     and self._g.is_initializer(n.input[0])
                                     and self._g.is_initializer(n.input[1]))]
        for reshape_op in constable_reshape_ops:
            target_t = numpy_helper.to_array(self._g.get_initializer(reshape_op.input[0]))
            target_shape = numpy_helper.to_array(self._g.get_initializer(reshape_op.input[1]))
            new_data = np.reshape(target_t, tuple(target_shape))
            const_name = utils.port_name(utils.make_name("Const"))
            new_tensor = numpy_helper.from_array(new_data, const_name)

            # point all children nodes inputs to the new node
            for output_name in reshape_op.output:
                for child in ops:
                    for i, name in enumerate(child.input):
                        if name == output_name:
                            child.input[i] = const_name
            self._g.add_initializer(new_tensor)
            # need call this to make input update synced to protobuf val
            self._g.update_proto()
            ops.remove(reshape_op)
            self._g.set_nodes(ops)
            self._g.topological_sort(ops)

    def post_optimize_action(self):
        nodes = self.nodes
        # if channel==1 or height==width==1, replace transpose with reshape
        for op in nodes:
            if op.type == "Transpose":
                input_shape = self._g.get_shape(op.input[0])
                if not input_shape:
                    continue

                new_shape = []
                # when transpose is NHWC_TO_NCHW
                if is_nchw_transpose(op) and (input_shape[3] == 1 or (input_shape[1] == 1 and input_shape[2] == 1)):
                    new_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
                # when transpose is NCHW_TO_NHWC
                if is_nhwc_transpose(op) and (input_shape[1] == 1 or (input_shape[2] == 1 and input_shape[3] == 1)):
                    new_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
                if new_shape:
                    out_nodes = self._g.find_output_consumers(op.output[0])
                    need_insert_reshape = False
                    for out_node in out_nodes:
                        if out_node.type != "Reshape":
                            need_insert_reshape = True
                    if need_insert_reshape:
                        op_name = utils.make_name("reshape")
                        shape_name = utils.make_name(op_name)
                        self._g.make_const(shape_name, np.array(new_shape, dtype=np.int64))
                        reshape_node = self._g.make_node("Reshape", inputs=[op.input[0], shape_name], outputs=op.output,
                                                         name=op_name)
                        self._update_graph_nodes([reshape_node], [op], True)
                    else:
                        self._remove_useless_tranpose(op)
        self._g.update_proto()
        self._g.topological_sort(self._g.get_nodes())

    def optimize(self):
        previous_counter = self._g.dump_node_statistics()
        no_action = False
        iteration_cnt = 0
        while not no_action:
            no_action = True
            nodes = self.nodes
            self._force_stop = {}
            for n in nodes:
                if is_nhwc_transpose(n):
                    if self._handle_nhwc_tranpose(n):
                        no_action = False
                        iteration_cnt += 1
                        # need break, because handler may change nodes set, making the n stale object
                        # referencing already deleted elements
                        break

                if is_useless_transpose(n):
                    no_action = False
                    iteration_cnt += 1
                    self._remove_useless_tranpose(n)
                    break
            # for debugging purpose
            if "stop" in self._force_stop and self._force_stop["stop"] == 1:
                break

        log.debug("finish after " + str(iteration_cnt) + " iteration(s)")
        self.post_optimize_action()

        current_counter = self._g.dump_node_statistics()
        transpose_cnt = current_counter["Transpose"]
        current_counter.subtract(previous_counter)
        log.info(" %d transpose op(s) left, ops diff after transpose optimization: %s", transpose_cnt, current_counter)
        if transpose_cnt > 2:
            log.warning("please try add --fold_const to help remove more transpose")

    def _initialize_handlers(self):
        self._handler_map = {
            "Add": self._add_handler,
            "Concat": self._concat_handler,
            "Identity": self._identity_handler,
            "Max": self._maxmin_handler,
            "Min": self._maxmin_handler,
            "Mul": self._mul_handler,
            "Pad": self._pad_handler,
            "ReduceMean": self._reducemean_handler,
            "Relu": self._relu_handler,
            "Slice": self._slice_handler,
            "Split": self._split_handler,
            "Tanh": self._tanh_handler,
            "Transpose": self._transpose_handler,
        }

    # if there is nodes added, removed, or inputs changed, we need update the output_nodes/output_number etc.
    def _update_graph_nodes(self, nodes_to_extend, nodes_to_remove, has_input_changed=False):
        ops = self.nodes

        if nodes_to_remove:
            for n in nodes_to_remove:
                ops.remove(n)

        if nodes_to_extend:
            ops.extend(nodes_to_extend)

        if nodes_to_extend or nodes_to_remove or has_input_changed:
            self._g.set_nodes(ops)

    def _handle_node_having_branches(self, node):
        # create transpose pairs if some input are not.
        self._create_transpose_pairs_before_node(node)

        # make sure node's all input transpose all have only 1 consumer node,
        # otherwise, it would impact their other output nodes
        if self._transpose_has_single_consumer_node(node.inputs):
            self._create_transpose_pairs_after_node(node)
            to_remove = []

            input_transposes = node.inputs
            for n in input_transposes:
                n_input = n.input[0]
                assert len(n.output) == 1
                self._g.replace_all_inputs(self._g.get_nodes(), n.output[0], n_input)

                to_remove.append(n)

            assert len(node.output) == 1
            # currently we assume node only has 1 output, for cases where it is more than 1 for example Split
            # we need consider the fact that Split's multiple output will not always has data in NCHW/NHWC,
            # it might be a different shape.
            output_transposes = self._g.find_output_consumers(node.output[0])
            for n in output_transposes:
                n_input = n.input[0]
                assert len(n.output) == 1
                self._g.replace_all_inputs(self._g.get_nodes(), n.output[0], n_input)

                to_remove.append(n)

            self._update_graph_nodes(None, to_remove, True)
            return True

        log.debug("input transpose does not have single consumer, skipping...")
        return False

    # the assumption is: both node and trans have only 1 output
    def _switch_transpose_and_node(self, node, trans):
        if not self._transpose_has_single_consumer_node([trans]):
            return False

        input_index = 0
        for i in node.input:
            if i == trans.output[0]:
                break
            else:
                input_index += 1

        ops = self._g.get_nodes()
        self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
        node.input[input_index] = trans.input[0]
        trans.input[0] = utils.port_name(node.name)

        # need to transpose node shape in backward direction as well after switch
        # otherwise, reshape added in post_optimize_action may not work correctly
        shape = self._g.get_shape(node.output[0])
        if shape:
            # only nhwc transpose can reach here
            new_shape = [shape[i] for i in [0, 3, 1, 2]]
            self._g.set_shape(node.output[0], new_shape)

        self._g.set_nodes(ops)
        return True

    # if return value is True, then it means Transpose is handled as designed
    # otherwise, it means that we skip handling since it is not in our support set
    def _handle_nhwc_tranpose(self, trans):
        out_nodes = self._g.find_output_consumers(trans.output[0])
        if len(out_nodes) == 1:
            p = out_nodes[0]
            if p.name in self._output_names:
                log.debug("cannot move transpose down since it met output node %s", p.name)
                return False

            if p.type in self._handler_map:
                op_handler = self._handler_map[p.type]
                return op_handler(trans, p)
            return False
        # move transpose into branches to let Transposes can be "handled" in each branch
        to_append = []
        for n in out_nodes:
            branch_trans = self._g.make_node("Transpose", [trans.input[0]], attr=trans.attr_onnx)
            self._g.replace_input(n, trans.output[0], branch_trans.output[0])

            to_append.append(branch_trans)
        self._update_graph_nodes(to_append, [trans], True)
        return False

    def _remove_useless_tranpose(self, trans):
        self._g.replace_all_inputs(self._g.get_nodes(), trans.output[0], trans.input[0])
        self._update_graph_nodes(None, [trans], True)

    def _transpose_has_single_consumer_node(self, trans_nodes):
        result = True
        for n in trans_nodes:
            cnt = len(set(self._g.find_output_consumers(n.output[0])))
            result = result and cnt == 1
            if not result:
                return False
        return True

    def _get_non_nchw_transpose_output_nodes(self, node):
        # we just support node having 1 output, we need consider cases where node has more than 1 outputs
        assert len(node.output) == 1
        non_nchw_tranpose_nodes = []
        consumers = self._g.find_output_consumers(node.output[0])
        for o in consumers:
            if not is_nchw_transpose(o) and o not in non_nchw_tranpose_nodes:
                non_nchw_tranpose_nodes.append(o)
        return non_nchw_tranpose_nodes

    def _create_transpose_pairs_after_node(self, node):
        assert len(node.output) == 1  # just support node who has 1 output
        non_nchw_trans_consumers = self._get_non_nchw_transpose_output_nodes(node)
        added_node = []
        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nchw_trans_consumers
        for consumer in non_nchw_trans_consumers:
            nchw_node = self._g.make_node("Transpose", [node.output[0]], attr={"perm": [0, 3, 1, 2]})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": [0, 2, 3, 1]})
            self._g.replace_input(consumer, node.output[0], nhwc_node.output[0])
            added_node.extend([nchw_node, nhwc_node])

        if added_node:
            self._update_graph_nodes(added_node, None, True)
        return added_node

    def _create_transpose_pairs_before_node(self, node):
        non_nhwc_trans_inputs = []
        for input_id, n in zip(node.input, node.inputs):
            if not is_nhwc_transpose(n):
                # check in case node has two inputs coming from a same node output.
                if [input_id, n] not in non_nhwc_trans_inputs:
                    non_nhwc_trans_inputs.append([input_id, n])

        added_node = []
        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nhwc_trans_consumers
        for input_id, n in non_nhwc_trans_inputs:
            nchw_node = self._g.make_node("Transpose", [input_id], attr={"perm": [0, 3, 1, 2]})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": [0, 2, 3, 1]})
            self._g.replace_input(node, input_id, nhwc_node.output[0])
            added_node.extend([nchw_node, nhwc_node])

        if added_node:
            self._update_graph_nodes(added_node, None, True)
        return added_node

    def _add_handler(self, trans, node):
        if self._g.is_initializer(node.input[1]):
            t_p = trans.inputs[0]
            if t_p.type in ("Conv", "ConvTranspose") and len(t_p.input) == 2:
                # if Conv or ConvTranspose's bias input is not set, then we set, otherwise, we don't set
                # todo: maybe we can add already set bias with the input??? try later
                conv_inputs = [t_p.input[0], t_p.input[1], node.input[1]]
                conv_node = self._g.make_node(t_p.type, conv_inputs, attr=t_p.attr_onnx)
                ops = self._g.get_nodes()
                trans.input[0] = utils.port_name(conv_node.name)
                self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
                self._update_graph_nodes([conv_node], [t_p, node], True)
                return True
            return False
        return self._handle_node_having_branches(node)

    def _relu_handler(self, trans, node):
        self._g.replace_all_inputs(self._g.get_nodes(), node.output[0], trans.output[0])
        node.input[0] = trans.input[0]
        trans.input[0] = node.output[0]
        return True

    def _transpose_handler(self, trans, node):
        if is_nchw_transpose(node):
            ops = self._g.get_nodes()
            self._g.replace_all_inputs(ops, node.output[0], trans.input[0])
            self._update_graph_nodes(None, [trans, node], True)
            return True
        return False

    def _maxmin_handler(self, trans, node):
        input_name = node.input[1]
        if self._g.is_initializer(input_name):
            numpy_val = numpy_helper.to_array(self._g.get_initializer(input_name))
            transposed_val = np.transpose(numpy_val, (0, 3, 1, 2))
            self._g.update_initializer(input_name, transposed_val)
            return self._switch_transpose_and_node(node, trans)
        return False

    def _mul_handler(self, trans, node):
        multiplier_input_id = None
        for i in node.input:
            if i != trans.output[0]:
                multiplier_input_id = i

        if not self._g.is_initializer(multiplier_input_id):
            return False
        multiplier = self._g.get_initializer(multiplier_input_id)

        # todo: apply this block if we have model case multiplier_input_id==0, and verify that.
        if multiplier_input_id == node.input[1]:
            t_p = trans.inputs[0]
            # make sure conv don't have bias set
            if t_p.type == "Conv" and self._g.is_initializer(t_p.input[1]) and len(t_p.input) == 2:
                conv = t_p
                numpy_val = numpy_helper.to_array(self._g.get_initializer(conv.input[1]))
                transposed_val = np.transpose(numpy_val, (2, 3, 1, 0))
                mul_val = numpy_helper.to_array(multiplier)
                result = np.multiply(transposed_val, mul_val)
                self._g.update_initializer(conv.input[1], np.transpose(result, (3, 2, 0, 1)))

                ops = self._g.get_nodes()
                self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
                self._update_graph_nodes(None, [node], True)
                return True

        # sometimes, dims is None as observed for a float scalar.
        # if there is only 1 number, so we just move transpose after the mul
        if not multiplier.dims or multiplier.dims[0] == 1:
            return self._switch_transpose_and_node(node, trans)

        return False

    def _identity_handler(self, trans, node):
        ops = self._g.get_nodes()
        self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
        self._update_graph_nodes(None, [node], True)
        return True

    def _concat_handler(self, trans, node):
        if self._handle_node_having_branches(node):
            node.set_attr("axis", 1)
            return True
        return False

    def _split_handler(self, trans, node):
        # Todo: need handle cases where Slit node has more than 1 outputs.
        if self._handle_node_having_branches(node):
            node.set_attr("axis", 1)
            return True
        return False

    def _pad_handler(self, trans, node):
        # [N-start, H-start, W-start, C-start, N-end, H-end,  W-end, C-end]
        pads = node.get_attr('pads').ints  # [x1_begin, x2_begin...x1_end, x2_end,...]
        # NHWC->NCHW
        new_pads = [pads[0], pads[3], pads[1], pads[2], pads[4], pads[7], pads[5], pads[6]]
        node.set_attr("pads", new_pads)
        return self._switch_transpose_and_node(node, trans)

    def _reducemean_handler(self, trans, node):
        axes = node.get_attr("axes").ints
        keepdims = node.get_attr("keepdims")
        # make sure keepdims is 1, then we can do the swap, otherwise, please don't, because
        # once keepdims is not set, original dims are lost, so transpose back won't work well.
        # by default, if keepdims is not specified, it is 1
        if axes == [1, 2] and ((keepdims and keepdims.i == 1) or (not keepdims)):
            node.set_attr("axes", [2, 3])
            return self._switch_transpose_and_node(node, trans)
        return False

    def _slice_handler(self, trans, node):
        axes = node.get_attr("axes").ints
        keepdims = node.get_attr("keepdims")
        if axes == [0, 1, 2, 3]:
            node.set_attr("axes", [0, 2, 3, 1])
            return self._switch_transpose_and_node(node, trans)
        return False

    # todo: consider share a same logic for element-wise op.
    def _tanh_handler(self, trans, node):
        return self._switch_transpose_and_node(node, trans)
