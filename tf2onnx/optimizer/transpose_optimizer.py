# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Transpose Optimizer."""

from __future__ import unicode_literals
from collections import defaultdict

import numpy as np

from tf2onnx import utils
from tf2onnx.optimizer.optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,abstract-method
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


class TransposeOptimizer(GraphOptimizerBase):
    """Transpose Optimizer."""

    def __init__(self):
        super(TransposeOptimizer, self).__init__()

        self._handler_map = {}
        self._force_stop = {}

        self._initialize_handlers()
        self._g = None
        self._output_names = None

    @property
    def nodes(self):
        return self._g.get_nodes()

    def pre_optimize_action(self):
        # make Reshape into a const, which then can be fused into Conv's weight for mobilenet_v1_75_192
        self._output_names = [name.split(":")[0] for name in self._g.outputs]
        ops = self.nodes
        constable_reshape_ops = [n for n in ops
                                 if (n.type == "Reshape"
                                     and n.inputs[0].is_const()
                                     and n.inputs[1].is_const())]
        for reshape_op in constable_reshape_ops:
            target_t = reshape_op.inputs[0].get_tensor_value(as_list=False)
            target_shape = reshape_op.inputs[1].get_tensor_value(as_list=False)
            new_data = np.reshape(target_t, tuple(target_shape))
            const_name = utils.port_name(utils.make_name("Const"))

            # point all children nodes inputs to the new node
            for output_name in reshape_op.output:
                for child in ops:
                    for i, name in enumerate(child.input):
                        if name == output_name:
                            child.input[i] = const_name
            self._g.make_const(const_name, new_data)
            self._g.remove_node(reshape_op.name)
            self._g.topological_sort(self._g.get_nodes())

    def post_optimize_action(self):
        nodes = self.nodes
        # if channel==1 or height==width==1, replace transpose with reshape
        for op in nodes:
            if op.type == "Transpose":
                input_shape = self._g.get_shape(op.input[0])
                if not input_shape:
                    continue
                # reshape only supports one dime is -1
                if input_shape.count(-1) > 1:
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
                        self._g.remove_node(op.name)
                        self._g.make_node("Reshape", inputs=[op.input[0], shape_name], outputs=op.output,
                                          name=op_name)
                    else:
                        self._remove_useless_tranpose(op)
        self._g.topological_sort(self._g.get_nodes())

    def merge_duplicated_transposes(self):
        # strategy used in previous procedure is to move transpose nodes down if possible,
        # and it means that when a node has n outputs then n transpose will be generated,
        # so we should merge them back to one if they can't be eliminated in previous procedure.
        graph = self._g
        input_transposes_map = defaultdict(list)
        for node in graph.get_nodes():
            if node.type == "Transpose" and node.get_attr("perm"):
                key = (node.input[0], str(node.get_attr("perm").ints))
                input_transposes_map[key].append(node)

        for transposes in input_transposes_map.values():
            # merge transpose nodes into one: make nodes use the output of the first transpose node
            transpose_out = transposes[0].output[0]
            for node in transposes[1:]:
                old_transpose_out = node.output[0]
                graph.replace_all_inputs(graph.get_nodes(), old_transpose_out, transpose_out)

        # dangling transpose nodes can be deleted
        graph.delete_unused_nodes(graph.outputs)

    def optimize(self, graph):
        self._g = graph
        self.pre_optimize_action()
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

        self.logger.debug("finish after " + str(iteration_cnt) + " iteration(s)")

        self.merge_duplicated_transposes()
        self.post_optimize_action()

        current_counter = self._g.dump_node_statistics()
        transpose_cnt = current_counter["Transpose"]
        self.logger.info(" %d transpose op(s) left", transpose_cnt)
        self._print_stat_diff(previous_counter, current_counter)
        if transpose_cnt > 2:
            self.logger.warning("please try add --fold_const to help remove more transpose")
        return self._g

    def _initialize_handlers(self):
        self._handler_map = {
            "Add": self._add_handler,
            "Cast": self._simple_through_handler,
            "Clip": self._simple_through_handler,
            "Concat": self._concat_handler,
            "Identity": self._identity_handler,
            "LeakyRelu": self._simple_through_handler,
            "Max": self._maxmin_handler,
            "Min": self._maxmin_handler,
            "Mul": self._mul_handler,
            "Pad": self._pad_handler,
            "ReduceMean": self._reducemean_handler,
            "Relu": self._simple_through_handler,
            "Shape": self._shape_handler,
            "Slice": self._slice_handler,
            "Split": self._split_handler,
            "Tanh": self._simple_through_handler,
            "Transpose": self._transpose_handler,
        }

    def _handle_node_having_branches(self, node):
        # create transpose pairs if some input are not.
        self._create_transpose_pairs_before_node(node)

        # make sure node's all input transpose all have only 1 consumer node,
        # otherwise, it would impact their other output nodes
        if self._transpose_has_single_consumer_node(node.inputs):
            self._create_transpose_pairs_after_node(node)
            input_transposes = node.inputs
            for n in input_transposes:
                n_input = n.input[0]
                utils.make_sure(len(n.output) == 1, "only expect single output")
                self._g.replace_all_inputs(self._g.get_nodes(), n.output[0], n_input)
                self._g.remove_node(n.name)

            utils.make_sure(len(node.output) == 1, "only expect single output")
            # currently we assume node only has 1 output, for cases where it is more than 1 for example Split
            # we need consider the fact that Split's multiple output will not always has data in NCHW/NHWC,
            # it might be a different shape.
            output_transposes = self._g.find_output_consumers(node.output[0])
            for n in output_transposes:
                n_input = n.input[0]
                utils.make_sure(len(n.output) == 1, "only expect single output")
                self._g.replace_all_inputs(self._g.get_nodes(), n.output[0], n_input)
                self._g.remove_node(n.name)
            return True

        self.logger.debug("input transpose does not have single consumer, skipping...")
        return False

    # get the input index of transpose op in node's inputs.
    def _get_input_index_for_trans(self, node, trans):
        input_index = 0
        for i in node.input:
            if i == trans.output[0]:
                break
            else:
                input_index += 1
        return input_index

    # the assumption is: both node and trans have only 1 output
    def _switch_transpose_and_node(self, node, trans):
        if not self._transpose_has_single_consumer_node([trans]):
            return False

        input_index = self._get_input_index_for_trans(node, trans)

        ops = self._g.get_nodes()
        self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
        node.input[input_index] = trans.input[0]
        trans.input[0] = node.output[0]

        # need to transpose node shape in backward direction as well after switch
        # otherwise, reshape added in post_optimize_action may not work correctly
        shape = self._g.get_shape(node.output[0])
        if shape:
            # only nhwc transpose can reach here
            new_shape = [shape[i] for i in [0, 3, 1, 2]]
            self._g.set_shape(node.output[0], new_shape)
        return True

    # if return value is True, then it means Transpose is handled as designed
    # otherwise, it means that we skip handling since it is not in our support set
    def _handle_nhwc_tranpose(self, trans):
        if trans.output[0] in self._g.outputs:
            self.logger.debug("%s connects to graph outputs, skip", trans.output[0])
            return False
        out_nodes = self._g.find_output_consumers(trans.output[0])
        if len(out_nodes) == 1:
            p = out_nodes[0]
            if p.name in self._output_names:
                self.logger.debug("cannot move transpose down since it met output node %s", p.name)
                return False

            if p.type in self._handler_map:
                op_handler = self._handler_map[p.type]
                return op_handler(trans, p)
            return False
        # move transpose into branches to let Transposes can be "handled" in each branch
        for n in out_nodes:
            branch_trans = self._g.make_node("Transpose", [trans.input[0]], attr=trans.attr_onnx)
            self._g.replace_input(n, trans.output[0], branch_trans.output[0])

        self._g.remove_node(trans.name)
        return False

    def _remove_useless_tranpose(self, trans):
        self._g.replace_all_inputs(self._g.get_nodes(), trans.output[0], trans.input[0])
        self._g.remove_node(trans.name)

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
        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nchw_trans_consumers
        for consumer in non_nchw_trans_consumers:
            nchw_node = self._g.make_node("Transpose", [node.output[0]], attr={"perm": [0, 3, 1, 2]})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": [0, 2, 3, 1]})
            self._g.replace_input(consumer, node.output[0], nhwc_node.output[0])

    def _create_transpose_pairs_before_node(self, node):
        non_nhwc_trans_inputs = []
        for input_id, n in zip(node.input, node.inputs):
            if not is_nhwc_transpose(n):
                # check in case node has two inputs coming from a same node output.
                if [input_id, n] not in non_nhwc_trans_inputs:
                    non_nhwc_trans_inputs.append([input_id, n])

        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nhwc_trans_consumers
        for input_id, n in non_nhwc_trans_inputs:
            nchw_node = self._g.make_node("Transpose", [input_id], attr={"perm": [0, 3, 1, 2]})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": [0, 2, 3, 1]})
            self._g.replace_input(node, input_id, nhwc_node.output[0])

    def _add_handler(self, trans, node):
        if node.inputs[1].is_const():
            t_p = trans.inputs[0]
            if t_p.type in ("Conv", "ConvTranspose") and len(t_p.input) == 2:
                # if Conv or ConvTranspose's bias input is not set, then we set, otherwise, we don't set
                # todo: maybe we can add already set bias with the input??? try later
                conv_inputs = [t_p.input[0], t_p.input[1], node.input[1]]
                conv_node = self._g.make_node(t_p.type, conv_inputs, attr=t_p.attr_onnx)
                ops = self._g.get_nodes()
                trans.input[0] = utils.port_name(conv_node.name)
                self._g.replace_all_inputs(ops, node.output[0], trans.output[0])

                self._g.remove_node(t_p.name)
                self._g.remove_node(node.name)
                return True
            return False
        return self._handle_node_having_branches(node)

    def _transpose_handler(self, trans, node):
        if is_nchw_transpose(node):
            ops = self._g.get_nodes()
            self._g.replace_all_inputs(ops, node.output[0], trans.input[0])

            shape = self._g.get_shape(node.output[0])
            dtype = self._g.get_dtype(node.output[0])
            self._g.remove_node(trans.name)
            self._g.remove_node(node.name)
            if node.output[0] in self._g.outputs:
                self._g.make_node("Identity", [trans.input[0]],
                                  outputs=node.output, shapes=[shape], dtypes=[dtype])
            return True
        return False

    def _maxmin_handler(self, trans, node):
        input_index = self._get_input_index_for_trans(node, trans)
        all_other_inputs = [input_id for i, input_id in enumerate(node.input) if i != input_index]

        all_other_inputs_const = all([self._g.get_node_by_output(i).is_const() for i in all_other_inputs])
        if all_other_inputs_const is False:
            return False

        shapes = [len(self._g.get_shape(i)) for i in all_other_inputs]
        shapes_not_one_and_four = [s for s in shapes if s not in [1, 4]]
        if shapes_not_one_and_four:
            return False

        for i in all_other_inputs:
            target_node = self._g.get_node_by_output(i)
            numpy_val = target_node.get_tensor_value(as_list=False)
            rank = numpy_val.ndim
            if rank == 4:
                transposed_val = np.transpose(numpy_val, (0, 3, 1, 2))
                target_node.set_tensor_value(transposed_val)
            elif rank == 1:  # scalar
                # do nothing
                pass
            else:
                raise ValueError("find rank !=1 and rank !=4, should not go here.")
        return self._switch_transpose_and_node(node, trans)

    def _mul_handler(self, trans, node):
        multiplier_input_id = None
        multiplier_input_node = None
        for i, input_node in zip(node.input, node.inputs):
            if i != trans.output[0]:
                multiplier_input_id = i
                multiplier_input_node = input_node

        # node's inputs may come from one same node. if so the multiplier_input_node may be none
        if multiplier_input_node is None or not multiplier_input_node.is_const():
            return False
        multiplier = multiplier_input_node.get_tensor_value(as_list=False)

        # todo: apply this block if we have model case multiplier_input_id==0, and verify that.
        if multiplier_input_id == node.input[1]:
            t_p = trans.inputs[0]
            # make sure conv don't have bias set
            if t_p.type == "Conv" and t_p.inputs[1].is_const() and len(t_p.input) == 2:
                conv = t_p
                numpy_val = conv.inputs[1].get_tensor_value(as_list=False)
                transposed_val = np.transpose(numpy_val, (2, 3, 1, 0))
                mul_val = multiplier
                result = np.multiply(transposed_val, mul_val)
                conv.inputs[1].set_tensor_value(np.transpose(result, (3, 2, 0, 1)))

                ops = self._g.get_nodes()
                self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
                self._g.remove_node(node.name)
                return True

        # if the shape is () or (1), we just move transpose after the mul
        if not multiplier.shape or (len(multiplier.shape) == 1 and multiplier.shape[0] == 1):
            return self._switch_transpose_and_node(node, trans)

        return False

    def _identity_handler(self, trans, node):
        if node.output[0] in self._g.outputs:
            return False
        ops = self._g.get_nodes()
        self._g.replace_all_inputs(ops, node.output[0], trans.output[0])
        self._g.remove_node(node.name)
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

    def _simple_through_handler(self, trans, node):
        return self._switch_transpose_and_node(node, trans)

    def _shape_handler(self, trans, node):
        # input > trans > shape  can be changed into  input > shape > gather
        if not self._transpose_has_single_consumer_node([trans]):
            return False

        output_shape = self._g.get_shape(node.output[0])
        output_dtype = self._g.get_dtype(node.output[0])
        self._g.remove_node(trans.name)
        self._g.remove_node(node.name)
        shape_node = self._g.make_node("Shape", [trans.input[0]])
        const_node = self._g.make_const("Const", np.array(trans.get_attr("perm").ints))
        gather_node = self._g.make_node("Gather", [shape_node.output[0], const_node.output[0]], outputs=node.output)
        self._g.set_shape(gather_node.output[0], output_shape)
        self._g.set_dtype(gather_node.output[0], output_dtype)
        return True
