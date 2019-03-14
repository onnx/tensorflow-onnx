# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter - generic loop support
"""

from __future__ import division
from __future__ import print_function
import logging
import sys
import traceback
from onnx import TensorProto
import numpy as np
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase, Context
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT
from tf2onnx.tfonnx import utils


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.loop_rewriter")


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access


class LoopRewriter(LoopRewriterBase):

    def create_context(self):
        return Context()

    def run(self):
        log.debug("enter loop rewriter")
        return self.run_internal()

    def need_rewrite(self, context):
        return True

    def rewrite(self, context):
        log.debug("enter rewrite function")
        loop_node = None
        try:
            loop_props = context.loop_properties
            cell_g_info = context.cell_graph
            cond_g_info = context.cond_graph

            # todo(pengwa): we don't check the case where loop body won't be executed at all.

            ## create Loop body graph with existing nodes

            # replace condition graph's inputs to be cell graph's outputs, because we want condition graph
            # to consumer cell graph outputs.
            for loop_var in cond_g_info.dependent_vars:
                self.g.replace_all_inputs(cond_g_info.nodes, loop_var.switch_true_identity_output.id,
                                          loop_var.next_iteration_input.id)

            body_nodes = set(cell_g_info.nodes + cond_g_info.nodes)
            body_outputs = cond_g_info.outputs + cell_g_info.outputs
            for out_tensor_value_info in body_outputs:
                out_tensor_value_info.shape = utils.create_vague_shape_like(out_tensor_value_info.shape)

            loop_body_g = LoopRewriterBase.construct_graph_from_nodes(self.g, body_nodes, body_outputs)

            # create loop body graph inputs
            loop_body_g.add_graph_input(utils.make_name("i"), TensorProto.INT64, ())
            loop_body_g.add_graph_input(utils.make_name("cond"), TensorProto.BOOL, ())
            for i, tensor_value_info in enumerate(loop_props.state_inputs):
                input_name = tensor_value_info.id
                if input_name is None:
                    # if the variable is not used in the body graph, then we created a fake one,
                    # the same type and shape as its corresponding output.
                    out_tensor_value_info = loop_props.state_outputs[i]
                    dtype = out_tensor_value_info.dtype
                    shape = out_tensor_value_info.shape
                    input_name = utils.make_name("unused_state_input_")
                else:
                    dtype = tensor_value_info.dtype
                    shape = tensor_value_info.shape

                loop_body_g.add_graph_input(input_name, dtype, utils.create_vague_shape_like(shape))

            for input_ta in loop_props.tensor_array_inputs:
                # Loop does not have scan inputs, so we use Gather to get data for each iteration.
                index_node = loop_body_g.make_node("Unsqueeze", [input_ta.index_input_id], attr={"axes": [0]})
                gather_node = loop_body_g.make_node("Gather", [input_ta.data_input_id, index_node.output[0]])
                data_node = loop_body_g.make_node("Squeeze", [gather_node.output[0]], attr={"axes": [0]})
                loop_body_g.replace_all_inputs(loop_body_g.get_nodes(), input_ta.consumer.id, data_node.output[0])

            ## create Loop node
            loop_node = self._create_loop_node(context, loop_props)
            if not loop_node:
                log.error("failed to create loop node during rewrite")
                return REWRITER_RESULT.FAIL
            loop_node.set_body_graph_as_attr("body", loop_body_g)

            log.debug("rewrite successfully")
            return REWRITER_RESULT.OK

        except Exception as ex:
            tb = traceback.format_exc()
            log.error("loop rewrite failed, due to exception: %s, details:%s", ex, tb)
            return REWRITER_RESULT.FAIL

    def _create_loop_node(self, context, loop_props):
        loop_outputs = []
        loop_output_shapes = []
        loop_output_dtypes = []
        for tensor_value_info in loop_props.state_outputs_exits + loop_props.scan_outputs_exits:
            if tensor_value_info.id:
                loop_outputs.append(tensor_value_info.id)
                loop_output_shapes.append(tensor_value_info.shape)
                loop_output_dtypes.append(tensor_value_info.dtype)
                n = self.g.get_node_by_output(tensor_value_info.id)
                self.g.remove_node(n.name)
            else:
                loop_outputs.append(utils.make_name("unused_loop_output_"))
                loop_output_shapes.append([-1])
                loop_output_dtypes.append(None)

        # trip count and cond are not used, giving them values just because bug
        # (https://github.com/Microsoft/onnxruntime/issues/255) of onnxruntime.
        trip_cnt = self.g.make_const(utils.make_name("trip_count"), np.array(sys.maxsize, dtype=np.int64))
        cond = self.g.make_const(utils.make_name("cond"), np.array(True, dtype=np.bool))
        loop_node = self.g.make_node("Loop", [trip_cnt.output[0]] + [cond.output[0]] +
                                     loop_props.state_inputs_initial_values,  # ONNX Loop support state inputs only
                                     outputs=loop_outputs, op_name_scope="generic_loop",
                                     shapes=loop_output_shapes, dtypes=loop_output_dtypes,
                                     skip_conversion=False)

        return loop_node
