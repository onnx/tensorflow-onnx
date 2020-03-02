# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Methods to load tensorflow graph from graphdef, checkpoint or saved_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from distutils.version import LooseVersion

import tensorflow as tf

from tf2onnx import utils
from tf2onnx.tf_utils import get_tf_version, tflist_to_onnx

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,unused-import,no-value-for-parameter,unexpected-keyword-arg,ungrouped-imports
# pylint: disable=missing-function-docstring,import-outside-toplevel,useless-import-alias,missing-docstring

def is_tf2():
    return tf.__version__.startswith("2.")


if is_tf2():
    from tensorflow.python.framework import convert_to_constants, func_graph, function_def_to_graph
else:
    from tensorflow.python.framework.graph_util import convert_variables_to_constants


if is_tf2():
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.compat.v1.GraphDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.io.gfile
    tf_placeholder = tf.compat.v1.placeholder
    extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
elif LooseVersion(tf.__version__) >= "1.13":
    # 1.13 introduced the compat namespace
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.compat.v1.GraphDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.compat.v1.placeholder
    extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
else:
    # older than 1.13
    tf_reset_default_graph = tf.reset_default_graph
    tf_global_variables = tf.global_variables
    tf_session = tf.Session  # pylint: disable=invalid-name
    tf_graphdef = tf.GraphDef
    tf_import_meta_graph = tf.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.placeholder
    extract_sub_graph = tf.graph_util.extract_sub_graph


def from_function(func, input_names, output_names):
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func, lower_control_flow=False)
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    # output_tensors = {i.name: i for i in frozen_func.outputs}
    tf_reset_default_graph()
    with tf_session() as sess:
        tf.import_graph_def(graph_def, name='')
        input_tensors = {i: sess.graph.get_tensor_by_name(i) for i in input_names}
        output_tensors = {i: sess.graph.get_tensor_by_name(i) for i in output_names}
        graph_def = tf_optimize(input_tensors, output_tensors, graph_def)
    return graph_def


def freeze_session(sess, input_names=None, output_names=None):
    """Freezes the state of a session into a pruned computation graph."""
    output_node_names = [i.split(':')[:-1][0] for i in output_names]
    keep_var_names = [i.split(':')[:-1][0] for i in input_names]
    with sess.graph.as_default():
        # freeze_var_names = list(set(v.op.name for v in tf_global_variables()).difference(keep_var_names or []))
        output_node_names = output_node_names or []
        output_node_names += [v.op.name for v in tf_global_variables()]
        output_node_names += keep_var_names
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        for node in graph_def.node:
            node.device = ""
        if is_tf2():
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
        else:
            graph_def = convert_variables_to_constants(sess, graph_def, output_node_names)
    return graph_def


def remove_redundant_inputs(frozen_graph, input_names):
    """Remove redundant inputs not in frozen graph."""
    frozen_inputs = []
    # get inputs in frozen graph
    for n in frozen_graph.node:
        for inp in input_names:
            if utils.node_name(inp) == n.name:
                frozen_inputs.append(inp)
    deleted_inputs = list(set(input_names) - set(frozen_inputs))
    if deleted_inputs:
        logger.warning("inputs [%s] is not in frozen graph, delete them", ",".join(deleted_inputs))
    return frozen_inputs


def from_graphdef(model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    with tf_session() as sess:
        graph_def = tf_graphdef()
        with tf_gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
        input_names = remove_redundant_inputs(frozen_graph, input_names)
        inputs = {i: sess.graph.get_tensor_by_name(i) for i in input_names}
        outputs = {i: sess.graph.get_tensor_by_name(i) for i in output_names}

    tf_reset_default_graph()
    with tf_session() as sess:
        frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
    tf_reset_default_graph()
    return frozen_graph, inputs, outputs


def from_checkpoint(model_path, input_names, output_names):
    """Load tensorflow graph from checkpoint."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    # model_path = checkpoint/checkpoint.meta
    with tf_session() as sess:
        saver = tf_import_meta_graph(model_path, clear_devices=True)
        # restore from model_path minus the ".meta"
        saver.restore(sess, model_path[:-5])
        frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
        input_names = remove_redundant_inputs(frozen_graph, input_names)
        inputs = {i: sess.graph.get_tensor_by_name(i) for i in input_names}
        outputs = {i: sess.graph.get_tensor_by_name(i) for i in output_names}

    tf_reset_default_graph()
    with tf_session() as sess:
        frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
    tf_reset_default_graph()
    return frozen_graph, inputs, outputs


def _from_saved_model_v1(sess, model_path, input_names, output_names, signatures):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    inputs = {}
    outputs = {}

    imported = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    for k in imported.signature_def.keys():
        if k.startswith("_"):
            # consider signatures starting with '_' private
            continue
        signatures.append(k)
    try:
        from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
        # pylint: disable=unnecessary-lambda
        get_signature_def = lambda meta_graph_def, k: \
            signature_def_utils.get_signature_def_by_key(meta_graph_def, k)
    except ImportError:
        # TF1.12 changed the api
        get_signature_def = lambda meta_graph_def, k: meta_graph_def.signature_def[k]

    for k in signatures:
        inputs_tensor_info = get_signature_def(imported, k).inputs
        for _, input_tensor in inputs_tensor_info.items():
            inputs[input_tensor.name] = sess.graph.get_tensor_by_name(input_tensor.name)
        outputs_tensor_info = get_signature_def(imported, k).outputs
        for _, output_tensor in outputs_tensor_info.items():
            outputs[output_tensor.name] = sess.graph.get_tensor_by_name(output_tensor.name)

    frozen_graph = freeze_session(sess, input_names=list(inputs.keys()), output_names=list(outputs.keys()))
    return frozen_graph, inputs, outputs


def _from_saved_model_v2(model_path, input_names, output_names, signatures):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    inputs = {}
    outputs = {}

    imported = tf.saved_model.load(model_path)  # pylint: disable=no-value-for-parameter

    #f = meta_graph_def.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    for k in imported.signatures.keys():
        if k.startswith("_"):
            # consider signatures starting with '_' private
            continue
        signatures.append(k)
    for k in signatures:
        concrete_func = imported.signatures[k]
        inputs = {input_tensor.name: input_tensor for input_tensor in concrete_func.inputs}
        outputs = {output_tensor.name: output_tensor for output_tensor in concrete_func.outputs}

    frozen_graph = from_function(concrete_func, list(inputs.keys()), list(outputs.keys()))
    return frozen_graph, inputs, outputs


def from_saved_model(model_path, input_names, output_names, signatures=None):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    if signatures is None:
        signatures = []
    tf_reset_default_graph()
    if is_tf2():
        frozen_graph, inputs, outputs = \
            _from_saved_model_v2(model_path, input_names, output_names, signatures)
        inputs = {k: v for k, v in inputs.items() if v.dtype != tf.dtypes.resource}
    else:
        with tf_session() as sess:
            frozen_graph, inputs, outputs = \
                _from_saved_model_v1(sess, model_path, input_names, output_names, signatures)

    if len(signatures) > 1:
        logger.warning("found multiple signatures %s in saved_model, pass --signature_def in command line",
                       signatures)

    tf_reset_default_graph()
    return frozen_graph, inputs, outputs


def from_keras(model_path, input_names, output_names):
    """Load keras model - experimental for now."""
    from tensorflow.python import keras as _keras
    from tensorflow.python.eager import context
    from tensorflow.python.keras.saving import saving_utils as _saving_utils

    # Handles Keras when Eager mode is enabled.
    custom_objects = None
    if context.executing_eagerly():
        _keras.backend.set_learning_phase(False)
        keras_model = _keras.models.load_model(model_path, custom_objects)

        function = _saving_utils.trace_model_call(keras_model)
        concrete_func = function.get_concrete_function()
        inputs = {input_tensor.name: input_tensor for input_tensor in concrete_func.inputs}
        outputs = {output_tensor.name: output_tensor for output_tensor in concrete_func.outputs}
        frozen_graph = from_function(concrete_func, list(inputs.keys()), list(outputs.keys()))
    else:
        # Handles Keras when Eager mode is disabled.
        _keras.backend.clear_session()
        _keras.backend.set_learning_phase(False)
        keras_model = _keras.models.load_model(model_path, custom_objects)
        sess = _keras.backend.get_session()
        frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
        inputs = {i: sess.graph.get_tensor_by_name(i) for i in keras_model.inputs}
        outputs = {i: sess.graph.get_tensor_by_name(i) for i in keras_model.outputs}
        tf_reset_default_graph()
        with tf_session() as sess:
            frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
        tf_reset_default_graph()
    return frozen_graph, inputs, outputs


def tf_optimize_grappler(input_tensors, output_tensors, graph_def, fold_constant=None):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    # don't use resource type as input
    output_tensors = list(output_tensors)

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    #   depends on so for now don't turn this on.
    rewrite_options.optimizers[:] = [
        # 'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
        'constfold', 'function'
    ]
    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in list(input_tensors) + output_tensors:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def tf_optimize(input_tensors, output_tensors, graph_def, fold_constant=True):
    """Extract inference subgraph and optimize graph."""
    assert isinstance(input_tensors, dict)
    assert isinstance(output_tensors, dict)
    try:
        input_tensors = {
            name: tensor for name, tensor in input_tensors.items()
            if tensor.dtype != tf.dtypes.resource
        }
    except:  # pylint: disable=bare-except
        pass

    # TODO: is this needed ?
    needed_names = [utils.node_name(i) for i in input_tensors.keys()] + \
               [utils.node_name(i) for i in output_tensors.keys()]
    graph_def = extract_sub_graph(graph_def, needed_names)

    if fold_constant:
        want_grappler = is_tf2() or LooseVersion(tf.__version__) >= "1.15"
        if want_grappler:
            graph_def = tf_optimize_grappler(input_tensors, output_tensors, graph_def, fold_constant)
        else:
            # the older transform path
            from tensorflow.tools.graph_transforms import TransformGraph  # pylint: disable=redefined-outer-name
            transforms = []
            if fold_constant:
                transforms.extend([
                    "fold_constants(ignore_errors=true)",
                    "remove_attribute(attribute_name=_class)",  # remove node colocation attributes
                ])
            transforms.extend([
                "fold_batch_norms",
                "fold_old_batch_norms",
            ])
            graph_def = TransformGraph(graph_def, input_tensors.keys(), output_tensors.keys(), transforms)

    return graph_def


def tf_reload_graph(tf_graph):
    """Invoke tensorflow cpp shape inference by reloading graph_def."""
    # invoke c api if tf version is below 1.8
    if get_tf_version() < LooseVersion("1.8"):
        logger.debug(
            "On TF < 1.8, graph is constructed by python API, " \
            "which doesn't invoke shape inference, please set " \
            "TF_C_API_GRAPH_CONSTRUCTION=1 to enable it"
        )

    graph_def = tf_graph.as_graph_def(add_shapes=True)
    with tf.Graph().as_default() as inferred_graph:
        tf.import_graph_def(graph_def, name="")
    return inferred_graph


def is_function(g):
    if is_tf2():
        return 'tensorflow.python.framework.func_graph.FuncGraph' in str(type(g))
    return False

_FUNCTIONS = {}


def resolve_functions(tf_graph):
    def toposort(data):
        while True:
            ordered = set(item for item, dep in data.items() if not dep)
            if not ordered:
                break
            yield ordered
            data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}

    _, _, _, _, _, functions = tflist_to_onnx(tf_graph, {})
    data = {}
    for k, fdef in tf_graph._functions.items():  # pylint: disable=protected-access
        input_shapes = functions.get(k)
        fdef = fdef.definition
        if input_shapes and len(fdef.signature.input_arg) < len(input_shapes):
            input_shapes = input_shapes[:len(fdef.signature.input_arg)]
        func = function_def_to_graph.function_def_to_graph(fdef, input_shapes=input_shapes)
        _FUNCTIONS[k] = func
        _, _, _, _, _, tfunctions = tflist_to_onnx(func, {})
        functions.update(tfunctions)
        data[k] = set(tfunctions.keys())

    result = []
    for d in toposort(data):
        result.extend(list(d))
    return [_FUNCTIONS[k] for k in result]


def set_function(name, func):
    _FUNCTIONS[name] = func


def find_function(name):
    return _FUNCTIONS.get(name)
