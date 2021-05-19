# SPDX-License-Identifier: Apache-2.0


"""
Dump onnx graph.
"""
# don't want to rename the tool
# pylint: disable=invalid-name

import argparse
import collections
import re

import onnx
from onnx import ModelProto
from onnx import helper, shape_inference


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


def main():
    args = get_args()

    with open(args.input, "rb") as f:
        data = f.read()
        model = ModelProto()
        model.ParseFromString(data)

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

    if args.check:
        onnx.checker.check_model(model)
        inferred_model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(inferred_model)

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
