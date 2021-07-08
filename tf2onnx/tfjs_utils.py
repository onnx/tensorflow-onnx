import json
import os
from onnx import numpy_helper, helper
import numpy as np
from tf2onnx import utils
import base64
from tf2onnx.graph import Graph
import gzip
from tensorflow.python.framework import c_api_util
from tensorflow.core.framework import types_pb2, node_def_pb2
from google.protobuf.json_format import ParseDict
import tensorflow as tf
from tf2onnx import tf_utils


tf_api_def_map = c_api_util.ApiDefMap()


def read_tfjs_attr_helper(k, v, tf_dtypes):
    utils.make_sure(k in ['func', 'shape', 'type', 'list', 's', 'i', 'f', 'b'], "TODO")
    if k == 'list':
        if len(v) == 0:
            return []
        else:
            utils.make_sure(len(v) == 1, "TODO")
            k2 = list(v.keys())[0]
            return [read_tfjs_attr_helper(k2, v2, tf_dtypes) for v2 in v[k2]]
    if k == 'type':
        dtype = getattr(types_pb2, v)
        if not tf_dtypes:
            dtype = tf_utils.map_tf_dtype(dtype)
        return dtype
    if k == 'func':
        return v['name']
    if k == 'shape':
        return [int(d['size']) for d in v.get('dim', [])]
    if k == 's':
        return base64.decodebytes(v.encode())
    if k == 'i':
        return int(v)
    return v


def read_tfjs_attr(attr, tf_dtypes=False):
    utils.make_sure(len(attr) == 1, "TODO")
    k = list(attr.keys())[0]
    return read_tfjs_attr_helper(k, attr[k], tf_dtypes)


def tfjs_node_to_tf_node_def(node):
    node_def = node_def_pb2.NodeDef()
    ParseDict(node, node_def)
    return node_def


def resolve_output(output, op_info):
    cnt = output.count(':')
    if cnt == 0:
        if output in op_info:
            return output + ':0'
        return output
    if cnt == 1:
        return output
    node, output_arg_name, port = output.split(':')
    op_type, attr = op_info[node]
    names, _ = get_output_names_and_dtypes(op_type, attr)
    idx = names.index(output_arg_name) + int(port)
    return node + ':' + str(idx)


def get_output_names_and_dtypes(op_type, attr):
    try:
        tf_op_def = tf_api_def_map.get_op_def(op_type)
    except ValueError:
        # TODO: fill this
        pass
    dtypes = []
    names = []
    for arg in tf_op_def.output_arg:
        num_copies = 1
        if arg.type_list_attr:
            dtypes += attr[arg.type_list_attr]
            num_copies = len(attr[arg.type_list_attr])
        else:
            if arg.type_attr:
                dtype = attr[arg.type_attr]
            else:
                dtype = arg.type
            if arg.number_attr:
                dtypes += [dtype] * attr[arg.number_attr]
                num_copies = attr[arg.number_attr]
            else:
                dtypes.append(dtype)
        names += [arg.name] * num_copies
    return names, dtypes


def get_output_shapes(node_def, input_dtypes, input_shapes, inp_consts):
    from tf2onnx.tf_loader import tf_session, tf_placeholder  # pylint: disable=import-outside-toplevel
    del node_def.input[:]
    node_def.name = "node"

    g = tf.Graph()
    with g.as_default():
        for i, (dtype, shape, const) in enumerate(zip(input_dtypes, input_shapes, inp_consts)):
            inp = "input" + str(i)
            if const is None:
                if shape is not None and -1 in shape:
                    shape = [d if d != -1 else None for d in shape]
                tf_placeholder(dtype, name=inp, shape=shape)
            else:
                tf.constant(const, dtype, name=inp)
            node_def.input.append(inp)
        mini_graph_def = g.as_graph_def()
        mini_graph_def.node.append(node_def)
    g2 = tf.Graph()
    with g2.as_default():
        with tf_session() as sess:
            tf.import_graph_def(mini_graph_def, name='')
            node = sess.graph.get_operation_by_name("node")
            outputs_shapes = [tf_utils.get_tf_tensor_shape(out) for out in node.outputs]
            return outputs_shapes


def graphs_from_tfjs(model_path, input_names=None, output_names=None):
    zip_compressed = False
    with open(model_path, "rb") as f:
        magic_number = f.read(2)
        f.seek(0)
        if magic_number == b'\x1f\x8b':
            unziped_bytes = gzip.decompress(f.read())
            model = json.loads(unziped_bytes)
            zip_compressed = True
        else:
            model = json.load(f)

    weightsManifest = model['weightsManifest'][0]

    sharded_data = []
    for path in weightsManifest["paths"]:
        with open(os.path.join(os.path.dirname(model_path), path), "rb") as f:
            shard_bytes = f.read()
            if zip_compressed:
                shard_bytes = gzip.decompress(shard_bytes)
            sharded_data.append(shard_bytes)

    tensor_data = b''.join(sharded_data)
    weights = {}

    i = 0
    for weight in weightsManifest['weights']:
        tensor_name = weight['name']
        count = np.product(weight['shape'], dtype=np.int64)
        np_dtype = np.dtype(weight['dtype'])
        if 'quantization' in weight:
            q_info = weight['quantization']
            q_dtype = np.dtype(q_info['dtype'])
            np_arr = np.frombuffer(tensor_data, dtype=q_dtype, count=count, offset=i)
            i += np_arr.nbytes
            np_arr = np_arr.astype(np_dtype) * q_info['scale'] + q_info['min']
        else:
            np_arr = np.frombuffer(tensor_data, dtype=np_dtype, count=count, offset=i)
            i += np_arr.nbytes
        np_arr = np_arr.reshape(weight['shape'])
        weights[tensor_name] = np_arr

    utils.make_sure(len(tensor_data) == i, "Total weight bytes %d doesn't match read bytes %d", len(tensor_data), i)
    topology = model['modelTopology']

    if output_names is None and 'signature' in model:
        output_names = [out for out in model['signature']['outputs'].keys()]

    main_g = read_tfjs_graph(topology['node'], weights, None, input_names, output_names)
    subgraphs = []
    for func in topology.get('library', {}).get('function', []):
        sub_g = read_tfjs_graph(func.get('nodeDef', []), weights, func)
        subgraphs.append(sub_g)

    return main_g, subgraphs


def read_tfjs_function(func):
    tf_dtypes = {}
    output_shapes = {}
    signature = func['signature']
    inputs = []
    for i, inp in enumerate(signature['inputArg']):
        inp_name = inp['name']
        inputs.append(inp_name)
        tf_dtypes[inp_name] = getattr(types_pb2, inp['type'])
        out_shapes_attr = func.get('argAttr', {}).get(str(i), {}).get('attr', {}).get('_output_shapes')
        if out_shapes_attr is not None:
            output_shapes[inp_name] = read_tfjs_attr(out_shapes_attr)[0]
    ret_map = func['ret']
    outputs = [ret_map[out['name']] for out in signature['outputArg']]
    name = signature['name']
    return tf_dtypes, output_shapes, inputs, outputs, name
            

def read_tfjs_graph(nodes, weights, func=None, graph_inputs=None, graph_outputs=None):
    onnx_nodes = []
    output_shapes = {}
    tf_dtypes = {}
    op_info = {}
    graph_name = 'tfjs_model'

    if func is not None:
        tf_dtypes, output_shapes, graph_inputs, graph_outputs, graph_name = read_tfjs_function(func)
        for inp in graph_inputs:
            onnx_nodes.append(helper.make_node("Placeholder", [], outputs=[inp], name=inp))

    # TODO: Placeholder with default, etc.
    if graph_inputs is None:
        graph_inputs = [n['name'] + ':0' for n in nodes if n['op'] == 'Placeholder']

    unused_outputs = set()

    for node in nodes:
        op_type = node['op']
        node_name = node['name']
        if op_type == "Const":
            np_arr = weights[node_name]
            out_name = node_name + ':0'
            onnx_tensor = numpy_helper.from_array(np_arr, out_name)
            onnx_node = helper.make_node("Const", [], outputs=[out_name], name=node_name, value=onnx_tensor)
            onnx_nodes.append(onnx_node)
            output_shapes[out_name] = np_arr.shape
            tf_dtypes[out_name] = read_tfjs_attr(node['attr']['dtype'], tf_dtypes=True)
            op_info[node_name] = (op_type, {'dtype': tf_dtypes[out_name]})
            continue
        tf_attr = {}
        onnx_attr = {}
        node_def = tfjs_node_to_tf_node_def(node)
        for k, v in node.get('attr', {}).items():
            tf_attr[k] = read_tfjs_attr(v, tf_dtypes=True)
            if k in tf_utils.TF_IGNORED_NODE_ATTRS:
                continue
            onnx_attr[k] = read_tfjs_attr(v)
        op_info[node_name] = (op_type, tf_attr)

        input_names = [resolve_output(inp, op_info) for inp in node.get('input', [])]
        unused_outputs.difference_update(input_names)
        inp_dtypes = [tf_dtypes[inp] for inp in input_names]
        inp_shapes = [output_shapes[inp] for inp in input_names]
        inp_consts = [weights.get(inp.split(':')[0]) for inp in input_names]
        _, out_dtypes = get_output_names_and_dtypes(op_type, tf_attr)
        out_shapes = get_output_shapes(node_def, inp_dtypes, inp_shapes, inp_consts)

        output_names = [node_name + ":" + str(i) for i in range(len(out_dtypes))]
        tf_dtypes.update(zip(output_names, out_dtypes))
        output_shapes.update(zip(output_names, out_shapes))
        unused_outputs.update(output_names)

        onnx_node = helper.make_node(op_type, input_names, output_names, name=node_name, **onnx_attr)
        onnx_nodes.append(onnx_node)

    dtypes = {k: tf_utils.map_tf_dtype(v) for k, v in tf_dtypes.items()}
    if graph_outputs is None:
        graph_outputs = list(unused_outputs)
    graph_outputs_mapped = [resolve_output(out, op_info) for out in graph_outputs]

    g = Graph(onnx_nodes, output_shapes, dtypes, input_names=graph_inputs, output_names=graph_outputs_mapped,
              is_subgraph=func is not None, graph_name=graph_name)
    g.rename_tensors(dict(zip(graph_outputs_mapped, graph_outputs)))
    return g
