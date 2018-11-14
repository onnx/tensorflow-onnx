# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Dump onnx graph.
"""
# don't want to rename the tool
# pylint: disable=invalid-name

from __future__ import division
from __future__ import print_function

import argparse
import collections
import re

import onnx
from onnx import ModelProto
from onnx import helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--pbtxt", help="write pbtxt")
    parser.add_argument("--dot", help="write dot file")
    parser.add_argument("--meta", help="include meta data", action="store_true")
    parser.add_argument("--check", help="check onnx model", action="store_true")
    parser.add_argument("--stats", help="collect stats", action="store_true")
    args = parser.parse_args()
    return args


def check_connectivity(model):
    """Check connectivity of all nodes."""
    g = model.graph
    inputs = set()
    outputs = set()
    # external_inputs = {i.name for i in g.input}
    external_outputs = {o.name for o in g.output}
    initializers = {i.name for i in g.initializer}
    for node in g.node:
        for i in node.input:
            inputs.add(i)
        for o in node.output:
            outputs.add(o)
    unconnected_inputs = initializers.union(outputs).difference(inputs)
    if unconnected_inputs:
        print("node inputs not connected: {}".format(unconnected_inputs))
    unused_outputs = outputs.union(outputs).difference(inputs.union(external_outputs))
    if unused_outputs:
        print("unused node outputs: {}".format(unused_outputs))
    unused_initializers = initializers.difference(inputs.union(external_outputs))
    if unused_initializers:
        print("unused initializers: {}".format(unused_initializers))


def main():
    args = get_args()

    with open(args.input, "rb") as f:
        data = f.read()
        model = ModelProto()
        model.ParseFromString(data)

    if args.check:
        check_connectivity(model)
        onnx.checker.check_model(model)

    if args.stats:
        ops = collections.Counter()
        for node in model.graph.node:
            ops[node.op_type] += 1
        print(ops, "\n\n")

    if args.meta:
        fields = ["ir_version", "producer_name", "producer_version", "name", "opset_import"]
        for name in fields:
            value = getattr(model, name, None)
            if value:
                print("{} = {}".format(name, value))
        for i in model.metadata_props:
            print("meta.{} = {}", i.key, i.value)

    print(helper.printable_graph(model.graph))

    if args.pbtxt:
        with open(args.pbtxt, "w") as f:
            f.write(str(model.graph))

    if args.dot:
        with open(args.dot, "w") as f:
            f.write("digraph graphname {\n")
            for node in model.graph.node:
                output_name = node.name
                name = node.name
                color = ""
                if node.op_type.startswith("_"):
                    color = ' color="yellow"'
                if node.op_type == "CELL":
                    color = ' color="red"'
                f.write('"{}" [label="{},{}"{}];\n'.format(output_name, node.op_type, name, color))
                for input_name in node.input:
                    parts = input_name.split(":")
                    input_name = re.sub(r"^\^", "", parts[0])
                    f.write('  "{}" -> "{}";\n'.format(input_name, output_name))
            f.write("}\n")


if __name__ == "__main__":
    main()
