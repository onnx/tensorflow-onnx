# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - rewrite tensorflow graph to onnx graph
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import sys
import traceback

import numpy as np
from onnx import onnx_pb

import tf2onnx
import tf2onnx.onnx_opset  # pylint: disable=unused-import
import tf2onnx.custom_opsets  # pylint: disable=unused-import
from tf2onnx.graph import Graph
from tf2onnx.rewriter import *  # pylint: disable=wildcard-import
from tf2onnx.shape_inference import infer_shape
from tf2onnx.tf_loader import is_function, resolve_functions, set_function
from tf2onnx.tf_utils import tensorflow_to_onnx, get_tf_version

from . import constants, logging, schemas, utils, handler

logger = logging.getLogger(__name__)


# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring
# pylint: disable=unused-variable


def rewrite_constant_fold(g, ops):
    """
    We call tensorflow transform with constant folding but in some cases tensorflow does
    fold all constants. Since there are a bunch of ops in onnx that use attributes where
    tensorflow has dynamic inputs, we badly want constant folding to work. For cases where
    tensorflow missed something, make another pass over the graph and fix want we care about.
    """
    func_map = {
        "Add": np.add,
        "GreaterEqual": np.greater_equal,
        "Cast": np.cast,
        "ConcatV2": np.concatenate,
        "Less": np.less,
        "ListDiff": np.setdiff1d,
        "Mul": np.multiply,
        "Pack": np.stack,
        "Range": np.arange,
        "Sqrt": np.sqrt,
        "Sub": np.subtract,
    }

    # pylint: disable=too-many-nested-blocks
    keep_looking = True
    while keep_looking:
        keep_looking = False
        for idx, op in enumerate(ops):
            func = func_map.get(op.type)
            if func is None: continue
            if set(op.output) & set(g.outputs): continue
            try:
                inputs = []
                for node in op.inputs:
                    if not node.is_const():
                        break
                    inputs.append(node.get_tensor_value(as_list=False))

                logger.debug("op name %s, %s, %s", op.name, len(op.input), len(inputs))
                if inputs and len(op.input) == len(inputs):
                    logger.info("folding node type=%s, name=%s" % (op.type, op.name))
                    if op.type == "Cast":
                        dst = op.get_attr_int("to")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(dst)
                        val = np.cast[np_type](*inputs)
                    elif op.type == "ConcatV2":
                        axis = inputs[-1]
                        values = inputs[:-1]
                        val = func(tuple(values), axis)
                    elif op.type == "ListDiff":
                        out_type = op.get_attr_int("out_idx")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(out_type)
                        val = func(*inputs)
                        val = val.astype(np_type)
                    elif op.type in ["Pack"]:
                        # handle ops that need input array and axis
                        axis = op.get_attr_int("axis")
                        val = func(inputs, axis=axis)
                    elif op.type == "Range":
                        dtype = op.get_attr_int("Tidx")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(dtype)
                        val = func(*inputs, dtype=np_type)
                    else:
                        val = func(*inputs)

                    new_node_name = utils.make_name(op.name)
                    new_output_name = new_node_name
                    old_output_name = op.output[0]
                    old_node_name = op.name
                    logger.debug("create const node [%s] replacing [%s]", new_node_name, old_node_name)
                    ops[idx] = g.make_const(new_node_name, val)

                    logger.debug("replace old output [%s] with new output [%s]", old_output_name, new_output_name)
                    # need to re-write the consumers input name to use the const name
                    consumers = g.find_output_consumers(old_output_name)
                    if consumers:
                        for consumer in consumers:
                            g.replace_input(consumer, old_output_name, new_output_name)

                    # keep looking until there is nothing we can fold.
                    # We keep the graph in topological order so if we folded,
                    # the result might help a following op.
                    keep_looking = True
            except Exception as ex:
                tb = traceback.format_exc()  # pylint: disable=bare-except
                logger.info("exception: %s, details: %s", ex, tb)
                # ignore errors

        # pylint: enable=too-many-nested-blocks
    return ops


def rewrite_incomplete_type_support(g, ops, impacted_ops):
    """
    for ops that have inclomplete type support, insert casts.
    This is needed for some tensor ops in opset7 and for some ops in winml-rs5.
    It is not helping performance but better than the model not working at all.
    """
    ignored_input_index = {
        "Tile": [1],  # Tile's second input can only be int64
        "Where": [0],  # Where's first input is bool
    }
    new_ops = []
    org_ops = list(ops)
    for op in org_ops:
        if op.type in impacted_ops:
            cast_inserted = []
            output_dtype = None
            ignored_inputs = ignored_input_index.get(op.type)
            # insert casts on inputs if the runtime only supports float
            for i, input_node in enumerate(op.inputs):
                if ignored_inputs and i in ignored_inputs:
                    continue

                input_name = op.input[i]
                dtype = g.get_dtype(input_name)
                if dtype is None:
                    logger.warning("adding Cast for op %s (type is %s)' input: %s, dtype should not be None",
                                   op.name, op.type, input_name)

                if dtype != onnx_pb.TensorProto.FLOAT:
                    output_dtype = dtype
                    logger.debug("insert cast for node %s on input %s", op.name, input_name)
                    if input_node and input_node.type == "Cast" \
                            and len(g.find_output_consumers(input_node.output[0])) == 1:
                        input_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
                        g.set_dtype(input_name, onnx_pb.TensorProto.FLOAT)
                    else:
                        cast_node = g.insert_new_node_on_input(op, "Cast", input_name,
                                                               to=onnx_pb.TensorProto.FLOAT)
                        g.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
                        g.copy_shape(input_name, cast_node.output[0])
                        cast_inserted.append(cast_node)
            if output_dtype:
                # insert reverse cast if needed
                for output_name in op.output:
                    name = utils.make_name(op.name)
                    logger.debug("insert cast back for node %s on output %s [dtype=%s]", op.name, output_name,
                                 output_dtype)
                    output_cast = g.insert_new_node_on_output("Cast", output_name, name=name,
                                                              to=output_dtype)
                    g.set_dtype(output_cast.output[0], output_dtype)
                    g.copy_shape(output_name, output_cast.output[0])
                    cast_inserted.append(output_cast)

            if cast_inserted:
                new_ops.extend(cast_inserted)
        new_ops.append(op)
    return new_ops


def rewrite_incomplete_type_support_rs5(g, ops):
    return rewrite_incomplete_type_support(g, ops, ["Unsqueeze", "Mul", "Concat", "Slice", "Transpose"])


def rewrite_incomplete_type_support_rs6(g, ops):
    impacted_ops = [
        "Div",
        "IsNaN",
        "Max",
        "Min",
        "ReduceSum",
        "Slice",
        "Split",
        "Tile",
        "Transpose",
        "Where"
    ]
    # TODO: logic to insert cast has bug, not all inputs of one node need cast
    # for example, slice's input "starts" doesn't need it.
    if g.opset == 10:
        impacted_ops.remove("Slice")

    return rewrite_incomplete_type_support(g, ops, impacted_ops)


def tensorflow_onnx_mapping(g, ops_mapping):
    logger.verbose("Mapping TF node to ONNX node(s)")
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()
    exceptions = []

    ops = list(g.get_nodes())
    for node in ops:
        logger.debug("Process node: %s\n%s", node.name, node.summary)

        if node.need_skip():
            logger.debug("explicitly skip node " + node.name)
            continue

        op = node.type
        map_info = ops_mapping.get(op)
        if map_info is None:
            unmapped_op[op] += 1
            logger.error("Tensorflow op [%s: %s] is not supported", node.name, op)
            continue
        mapped_op[op] += 1

        func, kwargs = map_info
        if kwargs:
            # if there is a onnx_op key we'll map the old type to a new type
            onnx_op = kwargs.get("onnx_op")
            if onnx_op:
                kwargs["tf_op"] = op
                node.type = onnx_op
        body_graphs = node.get_body_graphs()
        if body_graphs:
            for attr, b_g in body_graphs.items():
                logger.debug("start handling subgraph of %s's attribute %s", node.name, attr)
                b_g.topological_sort(b_g.get_nodes())
                # we assume only ONNX nodes have subgraph defined in pre-rewriters.
                # that means, if we create node having subgraphs in this step, the
                # created subgraphs' nodes won't be mapped.
                m_ops, unm_ops, body_exceptions = tensorflow_onnx_mapping(b_g, ops_mapping)
                mapped_op += m_ops
                unmapped_op += unm_ops
                # topological_sort on the body in case processing has changed the order
                b_g.topological_sort(b_g.get_nodes())
                exceptions.extend(body_exceptions)
                logger.debug("finish handling subgraph of %s's attribute %s", node.name, attr)

        try:
            func(g, node, **kwargs)
            node.skip_conversion = True
        except Exception as ex:
            logger.error("Failed to convert node %s\n%s", node.name, node.summary, exc_info=1)
            exceptions.append(ex)

    return mapped_op, unmapped_op, exceptions


def transpose_inputs(ctx, inputs_as_nchw):
    """Insert a transpose from NHWC to NCHW on model input on users request."""
    ops = []
    for node in ctx.get_nodes():
        for idx, output_name in enumerate(node.output):
            if output_name in inputs_as_nchw:
                shape = ctx.get_shape(output_name)
                if len(shape) != len(constants.NCHW_TO_NHWC):
                    logger.warning("transpose_input for %s: shape must be rank 4, ignored" % output_name)
                    ops.append(node)
                    continue
                # insert transpose
                op_name = utils.make_name(node.name)
                transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
                transpose.set_attr("perm", constants.NCHW_TO_NHWC)
                ctx.copy_shape(output_name, transpose.output[0])
                ctx.set_shape(output_name, np.array(shape)[constants.NHWC_TO_NCHW])
                ops.append(transpose)
                ops.append(node)
                continue
        ops.append(node)
    ctx.reset_nodes(ops)


def topological_sort(g, continue_on_error):
    ops = g.get_nodes()
    if not continue_on_error:
        g.topological_sort(ops)
    else:
        try:
            g.topological_sort(ops)
        except:  # pylint: disable=bare-except
            # if we continue on error, ignore graph cycles so we can report all missing ops
            pass


def run_rewriters(g, funcs, continue_on_error):
    """Rewrite the original graph and body graphs of nodes"""
    # NOTE(wayuanho):
    # 1. we don't sort graph here, rewriter is expected to do it on its own.
    # 2. the graph here may have circles, current topological_sort cannot handle it.
    for func in funcs:
        try:
            ops = func(g, g.get_nodes())
            g.reset_nodes(ops)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            logger.error("rewriter %s: exception %s", func, ex)
            ex_ext = traceback.format_exception(type_, value_, traceback_)
            if continue_on_error:
                logger.info(ex_ext)
            else:
                raise ex

        if utils.is_debug_mode():
            broken_outputs = g.check_integrity()
            if broken_outputs:
                logging.error(
                    "After rewriter %s, graph breaks at outputs %s",
                    func.__name__, broken_outputs
                )

    if g.contained_graphs:
        for dict_val in g.contained_graphs.values():
            for attr_name, b_g in dict_val.items():
                run_rewriters(b_g, funcs, attr_name)


def process_tf_graph(tf_graph, continue_on_error=False, verbose=False, target=None,
                     opset=None, custom_op_handlers=None, custom_rewriter=None,
                     extra_opset=None, shape_override=None, inputs_as_nchw=None,
                     input_names=None, output_names=None, is_subgraph=False):
    """Convert tensorflow graph to onnx graph.
        Args:
            tf_graph: tensorflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            verbose: print summary stats (deprecated)
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            custom_op_handlers: dictionary of custom ops handlers
            custom_rewriter: list of custom graph rewriters
            extra_opset: list of extra opset's, for example the opset's used by custom ops
            shape_override: dict with inputs that override the shapes given by tensorflow
            inputs_as_nchw: transpose inputs in list from nchw to nchw
            input_names: list of input node names in graph, input name format as node_name:port_id
            output_names: list of output node names in graph, output name format as node_name:port_id
        Return:
            onnx graph
    """
    if verbose:
        logger.warning("Argument verbose for process_tf_graph is deprecated. Please use --verbose option instead.")
    del verbose

    opset = utils.find_opset(opset)
    if not is_subgraph:
        logger.info("Using tensorflow=%s, onnx=%s, tf2onnx=%s/%s",
                    get_tf_version(), utils.get_onnx_version(), tf2onnx.__version__, tf2onnx.version.git_version[:6])
        logger.info("Using opset <onnx, %s>", opset)
        if opset > schemas.get_max_supported_opset_version():
            logger.warning("Currently installed onnx package %s is too low to support opset %s, "
                           "please upgrade onnx package to avoid potential conversion issue.",
                           utils.get_onnx_version(), opset)

    is_func = is_function(tf_graph)
    if not is_func:
        tf_graph = infer_shape(tf_graph, shape_override)

    if shape_override is None:
        shape_override = {}
    if inputs_as_nchw is None:
        inputs_as_nchw = []
    if target is None:
        target = constants.DEFAULT_TARGET

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, _ = tensorflow_to_onnx(tf_graph, shape_override)
    if not is_subgraph:
        # make tf2onnx internal subgraphs from the tensorflow subgraphs
        ordered_func = resolve_functions(tf_graph)
        for func in ordered_func:
            f_inputs_names = [t.name for t in func.inputs]
            f_output_names = [t.name for t in func.outputs]
            fg = process_tf_graph(func, continue_on_error, False, target, opset,
                                  custom_op_handlers, custom_rewriter,
                                  extra_opset, shape_override, inputs_as_nchw,
                                  f_inputs_names, f_output_names, is_subgraph=True)
            fg.graph_name = func.name
            fg.func_inputs = f_inputs_names
            set_function(func.name, fg)

    io_to_check = []
    if input_names:
        io_to_check.extend(input_names)
    if output_names:
        io_to_check.extend(output_names)

    if io_to_check:
        # check output existence in case user passed in wrong output ids
        non_exists = set(io_to_check) - set(output_shapes.keys())
        if non_exists:
            logger.error("\nFailed to convert: inputs/outputs specified do not exist, make sure your passed"
                         "in format: input/output_node_name:port_id. Problematical inputs/outputs are: %s \n",
                         non_exists)
            raise ValueError("Inputs/Outputs Not Found")

    g = Graph(onnx_nodes, output_shapes, dtypes, target, opset, extra_opset, output_names, is_subgraph=is_subgraph)

    # create ops mapping for the desired opsets
    ops_mapping = handler.tf_op.create_mapping(g.opset, g.extra_opset)

    # apply custom ops on top of the assembled opset. We can either complement the opset
    # or override existing ops with a custom op.
    if custom_op_handlers is not None:
        # below is a bit tricky since there are a few api's:
        # 1. the future way we want custom ops to be registered with the @tf_op decorator. THose handlers will be
        #     registered via the decorator on load of the module ... nothing is required here.
        # 2. the old custom op api: a dictionary of {name: (func, args[])
        #     We deal with this by using a compat_handler that wraps to old handler with a new style handler.
        #     This is tempoary to give people give to move to the new api and after tf2onnx-1.5 we want to remove this
        custom_opset = {}
        for k, v in custom_op_handlers.items():
            # FIXME: remove this after tf2onnx-1.5
            def compat_handler(ctx, node, **kwargs):
                # wrap old handler
                name = node.name
                args = kwargs["args"]
                func = kwargs["func"]
                return func(ctx, node, name, args)

            args = v[1]
            kwargs = {"func": v[0]}
            if args:
                onnx_op = args[0]
                kwargs["onnx_op"] = onnx_op
                args = args[1:]
            kwargs["args"] = args
            new_handler = handler.tf_op(k,
                                        domain=constants.TENSORFLOW_OPSET.domain,
                                        kwargs=kwargs)
            new_handler.register_compat_handler(compat_handler, 1)
            custom_opset[k] = (compat_handler, kwargs)
        ops_mapping.update(custom_opset)

    if inputs_as_nchw:
        transpose_inputs(g, inputs_as_nchw)

    # pre-processing graph rewrites
    # bi-directional re-writer should be placed after single directional re-writer
    rewriters = [rewrite_constant_fold, rewrite_quantize_and_dequantize, rewrite_transpose, rewrite_flatten,
                 rewrite_gemm, rewrite_random_uniform, rewrite_random_uniform_fold_const,
                 rewrite_random_normal, rewrite_dropout, rewrite_eye,
                 rewrite_leakyrelu, rewrite_thresholded_relu, rewrite_conv2d_with_pad,
                 rewrite_single_direction_lstm, rewrite_bi_direction_lstm,
                 rewrite_single_direction_gru, rewrite_bi_direction_gru,
                 rewrite_custom_rnn_cell, rewrite_generic_loop, rewrite_cond,
                 rewrite_biasadd_with_conv2d,
                 ]

    if custom_rewriter is not None:
        rewriters.extend(custom_rewriter)

    run_rewriters(g, rewriters, continue_on_error)

    # some nodes may already copied into inner Graph, so remove them from main Graph.
    g.delete_unused_nodes(output_names)
    topological_sort(g, continue_on_error)

    mapped_op, unmapped_op, exceptions = tensorflow_onnx_mapping(g, ops_mapping)
    if unmapped_op:
        logger.error("Unsupported ops: %s", unmapped_op)
    if exceptions and not continue_on_error:
        raise exceptions[0]

    # post-processing rewriters
    late_rewriters = []
    if constants.TARGET_RS5 in target:
        late_rewriters.append(rewrite_incomplete_type_support_rs5)
    if constants.TARGET_RS6 in target:
        late_rewriters.append(rewrite_incomplete_type_support_rs6)
    if late_rewriters:
        run_rewriters(g, late_rewriters, continue_on_error)

    # onnx requires topological sorting
    topological_sort(g, continue_on_error)

    g.update_proto()

    logger.verbose(
        "Summay Stats:\n"
        "\ttensorflow ops: {}\n"
        "\ttensorflow attr: {}\n"
        "\tonnx mapped: {}\n"
        "\tonnx unmapped: {}".format(op_cnt, attr_cnt, mapped_op, unmapped_op))

    return g


def tf_optimize(input_names, output_names, graph_def, fold_constant=True):
    """optimize tensorflow graph. This is in tf_loader but some apps call this
       so we proxy into tf_loader to keep them working."""
    return tf2onnx.tf_loader.tf_optimize(input_names, output_names, graph_def, fold_constant)
