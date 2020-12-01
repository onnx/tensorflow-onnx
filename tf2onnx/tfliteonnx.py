
from . import constants, logging, schemas, utils, handler
logger = logging.getLogger(__name__)

import tf2onnx
from tf2onnx.tf_utils import get_tf_version

def process_tf_lite(tflite_path, continue_on_error=False, target=None,
                    opset=None, custom_op_handlers=None, custom_rewriter=None,
                    extra_opset=None, shape_override=None, inputs_as_nchw=None,
                    is_subgraph=False):

    opset = utils.find_opset(opset)
    if not is_subgraph:
        logger.info("Using tensorflow=%s, onnx=%s, tf2onnx=%s/%s",
                    get_tf_version(), utils.get_onnx_version(), tf2onnx.__version__, tf2onnx.version.git_version[:6])
        logger.info("Using opset <onnx, %s>", opset)
        if opset > schemas.get_max_supported_opset_version():
            logger.warning("Currently installed onnx package %s is too low to support opset %s, "
                           "please upgrade onnx package to avoid potential conversion issue.",
                           utils.get_onnx_version(), opset)

    onnx_nodes, output_shapes, dtypes, _ = \
        tensorflow_to_onnx(tf_graph, shape_override, const_node_values)
    if not is_subgraph:
        # make tf2onnx internal subgraphs from the tensorflow subgraphs
        ordered_func = resolve_functions(tf_graph)
        for func in ordered_func:
            f_inputs_names = [t.name for t in func.inputs]
            f_output_names = [t.name for t in func.outputs]
            fg = process_tf_graph(func, continue_on_error, False, target, opset,
                                  custom_op_handlers, custom_rewriter,
                                  extra_opset, shape_override, inputs_as_nchw,
                                  f_inputs_names, f_output_names, is_subgraph=True,
                                  const_node_values=const_node_values)
            fg.graph_name = func.name
            fg.func_inputs = f_inputs_names
            set_function(func.name, fg)

    g = Graph(onnx_nodes, output_shapes, dtypes, target, opset, extra_opset, output_names, is_subgraph=is_subgraph)

    # create ops mapping for the desired opsets
    ops_mapping = handler.tf_op.create_mapping(g.opset, g.extra_opset)

    # apply custom ops on top of the assembled opset. We can either complement the opset
    # or override existing ops with a custom op.

    if inputs_as_nchw:
        transpose_inputs(g, inputs_as_nchw)

    # pre-processing graph rewrites
    # bi-directional re-writer should be placed after single directional re-writer
    rewriters = [
        # single directional
        rewrite_constant_fold,
        rewrite_quantize_and_dequantize,
        rewrite_transpose,
        rewrite_flatten,
        rewrite_random_uniform,
        rewrite_random_uniform_fold_const,
        rewrite_random_normal,
        rewrite_dropout,
        rewrite_eye,
        rewrite_leakyrelu,
        rewrite_thresholded_relu,
        rewrite_conv2d_with_pad,
        rewrite_single_direction_lstm,
        # bi-directional
        rewrite_bi_direction_lstm,
        rewrite_single_direction_gru,
        rewrite_bi_direction_gru,
        rewrite_custom_rnn_cell,
        rewrite_generic_loop, rewrite_cond,
        rewrite_biasadd_with_conv2d,
        rewrite_gemm,
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