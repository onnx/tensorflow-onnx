# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Dump onnx graph.
"""
import argparse
import collections
import onnx
from onnx import ModelProto
from onnx import helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--pbtxt", help="write pbtxt")
    parser.add_argument("--meta", help="include meta data", action="store_true")
    parser.add_argument("--check", help="check model", action="store_true")
    parser.add_argument("--stats", help="collect stats", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.input, "rb") as f:
        data = f.read()
        model = ModelProto()
        model.ParseFromString(data)

    if args.check:
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


if __name__ == "__main__":
    main()
