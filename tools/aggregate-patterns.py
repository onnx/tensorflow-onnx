# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Tool to find common patterns in onnx graphs.
"""

# don't want to rename the tool
# pylint: disable=invalid-name,missing-docstring,too-many-nested-blocks

from __future__ import division
from __future__ import print_function

import argparse
import collections
import csv
import os

from onnx import ModelProto


class Node():
    def __init__(self, node=None):
        self.inputs = []
        self.outputs = []
        self.name = ""
        self.op_type = ""
        if node:
            self.op_type = node.op_type
            self.name = node.name

    def add_input(self, node):
        self.inputs.append(node)

    def add_output(self, node):
        self.outputs.append(node)

    def walk(self, n):
        ret = []
        for node in self.inputs:
            if n > 0:
                next_node = node.walk(n - 1)
                if next_node:
                    ret.append(next_node)
            else:
                ret.append("*")
        if self.inputs:
            return self.op_type + "(" + ",".join(ret) + ")"
        return self.op_type + "()"

    def __str__(self):
        return "<node = " + self.op_type + "," + self.name + ">"

    def __repr__(self):
        return "<node = " + self.op_type + "," + self.name + ">"


class Graph():
    def __init__(self):
        self.nodes = []
        self.outputs = []

    def add(self, node):
        self.nodes.append(node)

    def add_output(self, node):
        self.outputs.append(node)

    def bfs(self, next_to_visit):
        nodes_to_keep = []
        while next_to_visit:
            n = next_to_visit[0]
            del next_to_visit[0]
            if n in nodes_to_keep:
                continue
            nodes_to_keep.append(n)
            for i in reversed(n.inputs):
                next_to_visit.append(i)
        return nodes_to_keep


def parse_onnx(fname):
    with open(fname, "rb") as f:
        data = f.read()
        model = ModelProto()
        model.ParseFromString(data)

    outputs = {}
    g = Graph()

    for node in model.graph.initializer:
        n = Node()
        n.op_type = "Const"
        n.name = node.name
        g.add(n)
        outputs[n.name] = n
    for node in model.graph.node:
        n = Node(node)
        for name in node.output:
            outputs[name] = n
        for name in node.input:
            o = outputs.get(name)
            if o:
                n.add_input(o)
        g.add(n)
    for node in model.graph.output:
        o = outputs.get(node.name)
        if o:
            g.add_output(o)
    return g


def one_file(fname, summary, max_nodes):
    g = parse_onnx(fname)
    for output_node in g.outputs:
        res = g.bfs([output_node])
        for node in res:
            ret = node.walk(max_nodes)
            if len(ret.split(",")) > 2:
                summary[ret] += 1


def read_csv(fname, summary):
    with open(fname, "r", encoding="utf-8") as fp:
        rd = csv.reader(fp)
        next(rd)
        for row in rd:
            if len(row) > 4:
                summary[row[4]] = int(row[1])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write to csv")
    parser.add_argument("--summary", help="write combined to csv")
    parser.add_argument("--fold", action="store_true", help="fold smaller sub graphs")
    parser.add_argument("--max-nodes", type=int, default=6, help="number of max nodes in a pattern")
    parser.add_argument("--min-frequency", type=int, default=2, help="show patterns number seen at least this")
    parser.add_argument("infile", nargs="*", help='event files')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.output:
        fp = open(args.output, "w")
        fp.write("model,seen,maxlength,length,pattern\n")

    summary_combined = collections.defaultdict(int)
    for fname in args.infile:
        summary = collections.defaultdict(int)
        for n in range(2, args.max_nodes):
            if fname.endswith(".onnx"):
                one_file(fname, summary, n)
                summary_sorted = sorted(summary.items(), key=lambda x: x[1], reverse=True)
                name = os.path.basename(fname)
                for k, v in summary_sorted:
                    if v > args.min_frequency:
                        print("{},{},{}".format(name, v, k))
                if args.output:
                    for k, v in summary_sorted:
                        if v > args.min_frequency:
                            l = len(k.split(","))
                            fp.write("{},{},{},{},\"{}\"\n".format(name, v, n, l, k))
            else:
                read_csv(fname, summary)

            for k, v in summary.items():
                summary_combined[k] += v

    if args.output:
        fp.close()

    if args.summary:
        summary_sorted = sorted(summary_combined.items(), key=lambda x: x[1], reverse=True)
        for k, v in summary_sorted:
            if v > args.min_frequency:
                print("combined,{},{}".format(v, k))
        with open(args.summary, "w") as fp:
            for k, v in summary_sorted:
                if v > args.min_frequency:
                    l = len(k.split(","))
                    fp.write("{},{},{},{},\"{}\"\n".format("combined", v, 0, l, k))


if __name__ == "__main__":
    main()
