# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
simple tool to convert .meta to .pb.
"""

from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", nargs="*", help='event files')
    args = parser.parse_args()
    return args


def to_pb(src):
    """Convert .meta to .pb."""
    _ = tf.train.import_meta_graph(src)
    graph = tf.get_default_graph()

    fname = os.path.basename(src)[:-5]
    tf.train.write_graph(graph, os.path.dirname(src), fname + '.pb', as_text=False)
    tf.train.write_graph(graph, os.path.dirname(src), fname + '.pbtxt', as_text=True)

    writer = tf.summary.FileWriter(os.path.dirname(src))
    writer.add_graph(graph)
    writer.close()


def main():
    args = get_args()
    for src in args.infile:
        to_pb(src)


if __name__ == "__main__":
    main()
