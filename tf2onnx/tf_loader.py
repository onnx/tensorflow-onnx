# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Methods to load tensorflow graph from graphdef, checkpoint or saved_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import tensorflow as tf
TF2 = tf.__version__.startswith("2.")

from tensorflow.python.framework import graph_util
if TF2:
    from tensorflow.python.framework import convert_to_constants
else:
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

from tf2onnx import utils

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument

if TF2:
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session
    tf_graphdef = tf.compat.v1.GraphDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.io.gfile
    tf_placeholder = tf.compat.v1.placeholder
else:
    # TODO: in theory we can just use tf.compat.v1 but currently
    # we support tf-1.4 and up. Need to see if we can limit support
    # to newer tensorflow version.
    # In the meantime we do it like below:
    tf_reset_default_graph = tf.reset_default_graph
    tf_global_variables = tf.global_variables
    tf_session = tf.Session
    tf_graphdef = tf.GraphDef
    tf_import_meta_graph = tf.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.placeholder


def freeze_func(func, outputs=None):
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func, lower_control_flow=True)
    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.dtypes.resource
    ]
    input_tensors = { i.name: i for i in input_tensors }

    if outputs:
        # below is important. tf.function will add and identity to the output in many cases and if
        # our output is also an identity, grappler will optimize our output away. Make sure to tell
        # grappler to preserve it.
        _, outputs = get_tensors_for_names(frozen_func.graph.as_graph_def(), [], outputs)
        output_tensors = {k: v for k, v in outputs.items()}
    else:
        # TODO: do we need frozen_func.outputs or can we just use our outputs ?
        output_tensors = { i.name: i for i in frozen_func.outputs }

    graph_def = tf_optimize(input_tensors, output_tensors, frozen_func.graph.as_graph_def())
    return graph_def


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.split(':')[:-1][0] for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf_global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf_global_variables()]
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        if TF2:
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        else:
            frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


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


def get_tensors_for_names(graph_def, input_names, output_names):
    inputs = {}
    outputs = {}
    tf_reset_default_graph()
    with tf_session() as sess:
        tf.import_graph_def(graph_def, name='')
        for i in input_names:
            inputs[i] = sess.graph.get_tensor_by_name(i)
        for i in output_names:
            outputs[i] = sess.graph.get_tensor_by_name(i)
    return inputs, outputs


def from_graphdef(model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    with tf_session() as sess:
        graph_def = tf_graphdef()
        with tf_gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            frozen_graph = freeze_session(sess, output_names=output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)

    # get the input and output tensors
    inputs, outputs = get_tensors_for_names(frozen_graph, input_names, output_names)

    # clean up
    tf_reset_default_graph()
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
        frozen_graph = freeze_session(sess, output_names=output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)

    # get the input and output tensors
    inputs, outputs = get_tensors_for_names(frozen_graph, input_names, output_names)

    # clean up
    tf_reset_default_graph()
    frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
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

    frozen_graph = freeze_session(sess, output_names=list(outputs.keys()))

    return frozen_graph, inputs, outputs


def _from_saved_model_v2(model_path, input_names, output_names, signatures):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    inputs = {}
    outputs = {}

    imported = tf.saved_model.load(model_path)

    #f = meta_graph_def.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    for k in imported.signatures.keys():
        if k.startswith("_"):
            # consider signatures starting with '_' private
            continue
        signatures.append(k)
    for k in signatures:
        concrete_func = imported.signatures[k]
        for input_tensor in concrete_func.inputs:
            inputs[input_tensor.name] = input_tensor
        for output_tensor in  concrete_func.outputs:
            outputs[output_tensor.name] = output_tensor

    frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func, lower_control_flow=True)
    frozen_graph = frozen_func.graph.as_graph_def()
    return frozen_graph, inputs, outputs


def from_saved_model(model_path, input_names, output_names, signatures=None):
    """Load tensorflow graph from saved_model."""
    # make sure we start with clean default graph
    if signatures == None:
        signatures = []
    tf_reset_default_graph()
    if TF2:
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


    # if command line overrides inputs or outputs, filter the tensors here
    if input_names:
        inputs = {k: v for k, v in inputs.items() if k in input_names}
    if output_names:
        outputs = {k: v for k, v in outputs.items() if k in output_names}

    # clean up
    tf_reset_default_graph()
    frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
    return frozen_graph, inputs, outputs


def tf_optimize_grappler(input_tensors, output_tensors, graph_def, fold_constant=None):
    from tensorflow.python.grappler import tf_optimizer as tf_opt
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.core.protobuf import rewriter_config_pb2

    # don't use resource type as input
    output_tensors = list(output_tensors)

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.constant_folding = rewrite_options.ON

    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in list(input_tensors) + output_tensors:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def tf_optimize_v1(input_tensors, output_tensors, graph_def, fold_constant):
    """Optimize tensorflow graph for inference."""
    try:
        # this should work on newer tensorflow versions, like tf-1.13 and up
        graph_def = tf_optimize_grappler(input_tensors, output_tensors, graph_def, fold_constant)
        return graph_def
    except Exception as e:
        # older versions might fail
        pass

    # before that try the old try TransformGraph
    from tensorflow.tools.graph_transforms import TransformGraph

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
    needed_names = [utils.node_name(i) for i in input_tensors.keys()] + \
                   [utils.node_name(i) for i in output_tensors.keys()]
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, input_tensors.keys(), output_tensors.keys(), transforms)
    return graph_def


def tf_optimize_v2(input_tensors, output_tensors, graph_def, fold_constant):
    with tf_session() as sess:
        graph_def = tf_optimize_grappler(input_tensors, output_tensors, graph_def, fold_constant)

    if isinstance(input_tensors, list):
        needed_names = [utils.node_name(i) for i in input_tensors] + \
                   [utils.node_name(i) for i in output_tensors]
    else:
        needed_names = [utils.node_name(i) for i in input_tensors.keys()] + \
                   [utils.node_name(i) for i in output_tensors.keys()]
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    return graph_def


def tf_optimize(inputs, outputs, graph_def, fold_constant=None):
    if TF2:
        return tf_optimize_v2(inputs, outputs, graph_def, fold_constant)
    return tf_optimize_v1(inputs, outputs, graph_def, fold_constant)
