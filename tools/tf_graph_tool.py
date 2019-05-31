# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" Tool for common tf graph operations. """

from __future__ import division
from __future__ import print_function

import argparse
from collections import Counter
import logging
import os
import sys
import copy

from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2

# pylint: disable=missing-docstring

logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)


def get_file_name(path):
    return os.path.basename(path)


def get_file_name_without_ext(path):
    return '.'.join(get_file_name(path).split('.')[:-1])


def replace_file_extension(path, ext):
    tokens = path.split('.')[:-1]
    tokens.append(ext)
    return '.'.join(tokens)


def append_file_name_suffix(path, suffix):
    tokens = path.split('.')
    tokens[-2] += '_' + suffix
    return '.'.join(tokens)


def get_file_directory(path):
    return os.path.dirname(path)


def get_file_directory_name(path):
    return os.path.basename(get_file_directory(path))


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_graph_def_from_pb(path):
    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    with open(path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def save_graph_def(graph_def, path, as_text=False):
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(graph_def))
    else:
        with open(path, "wb") as f:
            f.write(graph_def.SerializeToString())


def get_node_name(tensor_name):
    if tensor_name.startswith("^"):
        return tensor_name[1:]
    return tensor_name.split(":")[0]


def get_node_shape(node):
    shape_attr = node.attr.get("shape")
    shape = [d.size for d in shape_attr.shape.dim]
    return shape


def get_graph_def_io_nodes(graph_def):
    consumed = set()
    inputs = []
    outputs = []
    input_shapes = []
    for node in graph_def.node:
        for i in node.input:
            consumed.add(get_node_name(i))
        if node.op in ["Placeholder", "PlaceholderWithDefault", "PlaceholderV2"]:
            inputs.append(node.name)
            shape = []
            try:
                shape = get_node_shape(node)
            except:  # pylint: disable=bare-except
                pass
            input_shapes.append(shape)

    for node in graph_def.node:
        if node.name not in consumed and node.name not in inputs:
            outputs.append(node.name)

    return inputs, outputs, input_shapes


class main(object):
    @staticmethod
    def convert_pb_to_pbtxt(input_path, output_path=None):
        if not output_path:
            output_path = replace_file_extension(input_path, "pbtxt")

        logging.info("load from %s", input_path)
        graph_def = load_graph_def_from_pb(input_path)

        logging.info("save to %s", output_path)
        save_graph_def(graph_def, output_path, as_text=True)

    @staticmethod
    def convert_pb_to_summary(input_path, output_dir=None, start_tensorboard=False, port=6006):
        if not output_dir:
            output_dir = input_path + ".summary"

        logging.info("load from %s", input_path)
        graph_def = load_graph_def_from_pb(input_path)

        logging.info("save to %s", output_dir)
        create_directory(output_dir)
        with tf.Session() as sess:
            tf.import_graph_def(graph_def, name=get_file_name_without_ext(input_path))
            train_writer = tf.summary.FileWriter(output_dir)
            train_writer.add_graph(sess.graph)
            train_writer.close()

        if start_tensorboard:
            logging.info("launch tensorboard")
            os.system("start tensorboard --logdir {} --port {}".format(output_dir, port))
            os.system("start http://localhost:{}".format(port))

    @staticmethod
    def get_graph_io_nodes(input_path):
        logging.info("load from %s", input_path)
        graph_def = load_graph_def_from_pb(input_path)
        inputs, outputs, input_shapes = get_graph_def_io_nodes(graph_def)
        logging.info("graph has:")
        logging.info("\t%s inputs:", len(inputs))
        for input_name, input_shape in zip(inputs, input_shapes):
            print("\"{}:0\": {}".format(input_name, input_shape))
        logging.info("\t%s (possible) outputs:", len(outputs))
        for output in outputs:
            print("- {}:0".format(output))

    @staticmethod
    def print_graph_stat(input_path):
        logging.info("load from %s", input_path)
        graph_def = load_graph_def_from_pb(input_path)

        op_stat = Counter()
        for node in graph_def.node:
            op_stat[node.op] += 1

        logging.info("graph stat:")
        for op, count in sorted(op_stat.items(), key=lambda x: x[0]):
            logging.info("\t%s = %s", op, count)

    @staticmethod
    def extract_sub_graph(input_path, dest_nodes=None, output_path=None,
                          src_nodes=None, name_prefix=""):
        """
        Extract the subgraph within the boundary defined by dest_nodes and src_nodes if name_prefix is provided
        or the subgraph comprising all nodes with name that starts with name_prefix.
        dest_nodes/src_nodes and name_prefix aren't compatible. You only need to supply one of them.
        """
        logging.info("load from %s", input_path)
        graph_def = load_graph_def_from_pb(input_path)
        logging.info("\ttotal node = %s", len(graph_def.node))

        if (dest_nodes or src_nodes) and name_prefix:
            raise RuntimeError("dest_nodes/src_nodes and name_prefix are incompatible.")
        if not name_prefix:
            if not dest_nodes:
                _, dest_nodes, _ = get_graph_def_io_nodes(graph_def)
        else:
            dest_nodes = []
            for node in graph_def.node:
                if node.name.startswith(name_prefix):
                    dest_nodes.append(node.name)
        if not src_nodes:
            src_nodes = []

        if not isinstance(dest_nodes, list):
            raise TypeError("dest_nodes must be a list.")
        if not isinstance(src_nodes, list):
            raise TypeError("src_nodes must be a list.")

        def extract_graph_summary(graph_def):
            """Extracts useful information from the graph and returns them."""
            name_to_input_name = {}  # Keyed by the dest node name.
            name_to_node = {}  # Keyed by node name.

            # Keeps track of node sequences. It is important to still output the
            # operations in the original order.
            name_to_seq_num = {}  # Keyed by node name.
            seq = 0
            for node in graph_def.node:
                n = get_node_name(node.name)
                name_to_node[n] = node
                name_to_input_name[n] = [get_node_name(x) for x in node.input]
                name_to_seq_num[n] = seq
                seq += 1
            return name_to_input_name, name_to_node, name_to_seq_num


        def assert_nodes_are_present(name_to_node, nodes):
            """Assert that nodes are present in the graph."""
            for d in nodes:
                assert d in name_to_node, "%s is not in graph" % d


        def bfs_for_reachable_nodes(target_nodes, name_to_input_name, checker=None):
            """Breadth first search for reachable nodes from target nodes."""
            nodes_to_keep = set()
            # Breadth first search to find all the nodes that we should keep.
            next_to_visit = target_nodes[:]
            while next_to_visit:
                n = next_to_visit[0]
                del next_to_visit[0]
                if n in nodes_to_keep:
                    # Already visited this node.
                    continue
                if not checker or checker(n):
                    nodes_to_keep.add(n)
                    next_to_visit += name_to_input_name[n]
            return nodes_to_keep

        name_to_input_name, name_to_node, name_to_seq_num = extract_graph_summary(
            graph_def)
        assert_nodes_are_present(name_to_node, dest_nodes)
        assert_nodes_are_present(name_to_node, src_nodes)

        src_ops = []
        def node_checker(n):
            if not n.startswith(name_prefix) or n in src_nodes:
                src_ops.append(name_to_node[n])
                return False
            return True
        nodes_to_keep = bfs_for_reachable_nodes(dest_nodes, name_to_input_name, checker=node_checker)

        nodes_to_keep_list = sorted(
            list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
        # Now construct the output GraphDef
        out = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
            out.node.extend([copy.deepcopy(name_to_node[n])])

        # create placeholder
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        for op in src_ops:
            placeholder_node = node_def_pb2.NodeDef()
            placeholder_node.op = "Placeholder"
            placeholder_node.name = op.name
            if str(op.attr["dtype"]):
                placeholder_node.attr["dtype"].CopyFrom(op.attr["dtype"])
            elif str(op.attr["T"]):
                placeholder_node.attr["dtype"].CopyFrom(op.attr["T"])
            shape = graph_util.tensor_shape_from_node_def_name(tf_graph, op.name)
            placeholder_node.attr["shape"].CopyFrom(
                attr_value_pb2.AttrValue(shape=shape.as_proto())
            )
            out.node.extend([placeholder_node])

        out.library.CopyFrom(graph_def.library)
        out.versions.CopyFrom(graph_def.versions)

        if not output_path:
            output_path = append_file_name_suffix(input_path, "sub")
        logging.info("save to %s", output_path)
        logging.info("\ttotal node = %s", len(out.node))
        save_graph_def(out, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # pb2txt
    subparser = subparsers.add_parser("pb2txt", help="convert pb to pbtxt")
    subparser.add_argument("--input", dest="input_path", required=True, help="input pb path")
    subparser.add_argument("--output", dest="output_path", help="output pbtxt path")
    subparser.set_defaults(func=main.convert_pb_to_pbtxt)

    # pb2summary
    subparser = subparsers.add_parser("pb2summary", help="create summary from pb")
    subparser.add_argument("--input", dest="input_path", required=True, help="input pb path")
    subparser.add_argument("--output", dest="output_dir", help="output summary directory")
    subparser.add_argument("--tb", dest="start_tensorboard", action="store_true", default=False,
                           help="open with tensorboard")
    subparser.add_argument("--port", type=int, help="tensorboard port")
    subparser.set_defaults(func=main.convert_pb_to_summary)

    # io
    subparser = subparsers.add_parser("io", help="get input nodes for graph, guess output nodes")
    subparser.add_argument("--input", dest="input_path", required=True, help="input pb path")
    subparser.set_defaults(func=main.get_graph_io_nodes)

    # stat
    subparser = subparsers.add_parser("stat", help="print stat")
    subparser.add_argument("--input", dest="input_path", required=True, help="input pb path")
    subparser.set_defaults(func=main.print_graph_stat)

    # extract
    subparser = subparsers.add_parser("extract", help="extract sub-graph")
    subparser.add_argument("--input", dest="input_path", required=True, help="input pb path")
    subparser.add_argument("--output", dest="output_path", help="output pb path")
    subparser.add_argument("--dest_nodes", help="dest nodes", default=None, action="append")
    subparser.add_argument("--src_nodes", help="source nodes", default=None, action="append")
    subparser.add_argument(
        "--name_prefix",
        help="prefix of name scope, incompatible with dest_nodes/src_nodes",
        default=None
    )
    subparser.set_defaults(func=main.extract_sub_graph)

    if len(sys.argv) <= 2:
        parser.print_help()
        sys.exit()

    (args, unknown) = parser.parse_known_args()

    func = args.func
    del args.func

    args = dict(filter(lambda x: x[1], vars(args).items()))
    func(**args)
