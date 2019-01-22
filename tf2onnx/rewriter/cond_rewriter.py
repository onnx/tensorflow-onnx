# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter_base
"""

from __future__ import division
from __future__ import print_function
import logging
from collections import defaultdict, OrderedDict
from tf2onnx import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.cond_rewriter_base")

# pylint: disable=missing-docstring, unused-argument

class CondGraphContext:
    """context for each branch graph"""
    def __init__(self):
        self.output = set()
        self.nodes = set()

    def empty(self):
        return len(self.output) == 0 and len(self.nodes) == 0

    def intersection(self, cond_body_graph):
        intersect_node = [
            n.name for n in self.nodes.intersection(cond_body_graph.nodes)
        ]
        intersect_output = self.output.intersection(cond_body_graph.output)
        return list(intersect_node) + list(intersect_output)

    def same_branch(self, graph_context):
        if not self.output or not graph_context.output:
            return False
        return list(self.output)[0] == list(graph_context.output)[0]

    def union(self, cond_body_graph):
        self.output |= cond_body_graph.output
        self.nodes |= cond_body_graph.nodes


class CondContext:
    def __init__(self, cond_scope, pred_input, true_graph_context,
                 false_graph_context, switchs, merges):
        self.cond_scope = cond_scope # name scope for this tf.cond
        self.pred_input = pred_input # condition input
        self.true_graph_context = true_graph_context
        self.false_graph_context = false_graph_context
        self.switchs = set(switchs)
        self.merges = merges # list of merges in order
        self.if_node = None


class CondRewriter:
    def __init__(self, g):
        self.g = g

    def rewrite(self):
        log.debug("enter cond pre rewrite")
        return self.run()

    def run(self):
        """tf.cond rewriter"""
        # parse tf.cond in topological sort order.
        # NOTE: we assume the current graph is a DAG.
        name_scope_merges = OrderedDict()
        self.g.topological_sort(self.g.get_nodes())
        all_nodes = self.g.get_nodes()
        for n in all_nodes:
            if self._is_merge(n):
                name_scope = utils.tf_name_scope(n.name)
                if name_scope not in name_scope_merges:
                    name_scope_merges[name_scope] = []
                name_scope_merges[name_scope].append(n)
        # check if need rewrite
        if not name_scope_merges.keys():
            return all_nodes

        for name_scope, merge_nodes in name_scope_merges.items():
            pred_input, true_graph_context, false_graph_context, switchs = \
                self._parse_cond(name_scope, merge_nodes)
            cond_context = CondContext(
                name_scope,
                pred_input,
                true_graph_context,
                false_graph_context,
                switchs,
                merge_nodes
            )
            nodes_to_add, nodes_to_remove = self._cut_off_connection(cond_context)
            self._set_branch_graph(cond_context)
            nodes_to_remove.extend(
                list(cond_context.true_graph_context.nodes) + \
                list(cond_context.false_graph_context.nodes)
            )
            for n in nodes_to_remove:
                if n in all_nodes:
                    all_nodes.remove(n)
            all_nodes.extend(nodes_to_add)
            self.g.set_nodes(all_nodes)
        log.debug("cond pre rewrite done")

        return self.g.get_nodes()

    def _cut_off_connection(self, cond_context):
        """cut off switchs and merges"""
        nodes_to_remove = list(cond_context.switchs) + list(cond_context.merges)
        nodes_to_add = []
        log.debug("cut off switch connection")
        # replace switch with identity node
        for switch in cond_context.switchs:
            false_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[0]],
                op_name_scope=cond_context.cond_scope
            )
            cond_context.false_graph_context.nodes.add(false_switch_id)
            true_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[1]],
                op_name_scope=cond_context.cond_scope
            )
            cond_context.true_graph_context.nodes.add(true_switch_id)
            nodes_to_add.extend([false_switch_id, true_switch_id])
        # replace merge with if node
        log.debug("cut off merge connection")
        cond_context.if_node = self.g.make_node(
            "If",
            [cond_context.pred_input],
            op_name_scope=cond_context.cond_scope,
            outputs=[m.output[0] for m in cond_context.merges],
            skip_conversion=False
        )
        nodes_to_add.append(cond_context.if_node)
        return nodes_to_add, nodes_to_remove

    def _is_switch(self, node):
        return node.type == "Switch"

    def _is_merge(self, node):
        return node.type == "Merge"

    def _pair_output_with_merge(self, true_output, false_output, merges):
        """pair output according to the order of merges"""
        log.debug(
            "pair ture and false output according to the order of merge"
        )
        log.debug("true outuput: %s", true_output)
        log.debug("false output: %s", false_output)
        log.debug("merges: %s", [m.input for m in merges])
        paired_false_output = []
        paired_true_output = []
        for merge in merges:
            f_input = None
            if merge.input[0] in true_output:
                paired_true_output.append(merge.input[0])
                f_input = merge.input[1]
            elif merge.input[1] in true_output:
                paired_true_output.append(merge.input[1])
                f_input = merge.input[0]
            else:
                raise ValueError(
                    "No output info in true branch outputs to merge node: {}".format(merge.name)
                )
            if f_input in false_output:
                paired_false_output.append(f_input)
            else:
                raise ValueError(
                    "No output info in false branch outputs to merge node: {}".format(merge.name)
                )
        return paired_true_output, paired_false_output, merges

    def _parse_cond(self, name_scope, merge_nodes):
        """parse condition subgraph for these merge nodes"""
        true_graph_context, false_graph_context, switchs = self._trace_back(name_scope, merge_nodes)
        # find pred output from any switch
        pred_input = list(switchs)[0].input[1]
        return pred_input, true_graph_context, false_graph_context, switchs

    def _trace_back(self, name_scope, merge_nodes):
        """
        trace back to the switch from merge nodes and collect the nodes
        in the true/false branchs of tf.cond respectively, some comments:
        1. According to tf.cond implementation, We make the hypothesis
           that one tf.cond cannot comprise successive Switch nodes.
        2. This implement doesn't depend on control inputs. For a price,
           in the case that true and false branch both only contain a
           const node, we will throw a Exception.
        3. Thank to construct_graph_from_nodes, in which Identity node
           will add to each output of subgraph, we needn't deal with the
           branch with only one const node specially.
        """
        log.debug("trace back from [%s]", ",".join(n.name for n in merge_nodes))
        stack = [m for m in merge_nodes]
        downstream_graph = defaultdict(CondGraphContext)
        non_input_node_downstream_graph = []
        true_graph_context = CondGraphContext()
        false_graph_context = CondGraphContext()
        # take down output info
        for merge_node in merge_nodes:
            for i in merge_node.input:
                input_node = self.g.get_node_by_output(i)
                downstream_graph[input_node].output.add(i)
                # if switch connects to merge directly
                if self._is_switch(input_node):
                    if i == input_node.output[0]:
                        false_graph_context.output.add(i)
                    else:
                        true_graph_context.output.add(i)
        switchs = set()
        while stack:
            node = stack.pop()
            inputs = node.input + node.get_implicit_inputs()
            if not inputs:
                non_input_node_downstream_graph.append(
                    downstream_graph[node]
                )
            for inp in inputs:
                input_node = self.g.get_node_by_output(inp)
                if self._is_merge(input_node):
                    raise ValueError("nest merge at {} in {}".format(input_node.name, name_scope))
                # stop at the first switch
                if self._is_switch(input_node):
                    log.debug("encounter the first switch: %s", input_node.name)
                    # false branch
                    if input_node.output[0] == inp:
                        false_graph_context.union(downstream_graph[node])
                    # true branch
                    else:
                        true_graph_context.union(downstream_graph[node])
                    switchs.add(input_node)
                    # self._workaround_for_placeholder(input_node.input[0])
                else:
                    downstream_graph[input_node].nodes.add(input_node)
                    downstream_graph[input_node].union(downstream_graph[node])
                    stack.append(input_node)
        if true_graph_context.empty() and false_graph_context.empty():
            raise ValueError("Cannot handle the case both true and false branchs only \
                             contain const nodes for now.")
        for graph_context in non_input_node_downstream_graph:
            if graph_context.same_branch(true_graph_context) or \
                    true_graph_context.empty():
                true_graph_context.union(graph_context)
            else:
                false_graph_context.union(graph_context)
        # one node cannot belong to both true and false graph
        intersection = true_graph_context.intersection(false_graph_context)
        if intersection:
            raise ValueError("true graph and false graph intersect at [{}]".format(
                ",".join(intersection)
            ))
        log.debug("=================false body graph===============")
        log.debug(false_graph_context.nodes)
        log.debug(false_graph_context.output)
        log.debug("=================true body graph===============")
        log.debug(true_graph_context.nodes)
        log.debug(true_graph_context.output)
        return true_graph_context, false_graph_context, switchs

    def _set_branch_graph(self, cond_context):
        """set body graph for each branch"""
        log.debug("set graph for if branchs")
        paired_true_output, paired_false_output, _ = self._pair_output_with_merge(
            cond_context.true_graph_context.output,
            cond_context.false_graph_context.output,
            cond_context.merges
        )
        true_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.true_graph_context.nodes),
            paired_true_output,
            [self.g.get_shape(out) for out in paired_true_output],
            [self.g.get_dtype(out) for out in paired_true_output]
        )
        false_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.false_graph_context.nodes),
            paired_false_output,
            [self.g.get_shape(out) for out in paired_false_output],
            [self.g.get_dtype(out) for out in paired_false_output]
        )
        cond_context.if_node.set_body_graph_as_attr("then_branch", true_graph)
        cond_context.if_node.set_body_graph_as_attr("else_branch", false_graph)


def rewrite_cond(g, ops):
    return CondRewriter(g).rewrite()
