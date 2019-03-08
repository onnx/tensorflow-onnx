# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Make simple test model in all tensorflow formats."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# pylint: disable=missing-docstring

# Parameters
learning_rate = 0.02
training_epochs = 100

# Training Data
_train_x = np.array(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
_train_y = np.array(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
_test_x = np.array([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
_test_y = np.array([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.replace(":0", "") for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def train(model_path):
    n_samples = _train_x.shape[0]

    # tf Graph Input
    x = tf.placeholder(tf.float32, name="X")
    y = tf.placeholder(tf.float32, name="Y")

    # Set model weights
    w = tf.Variable(np.random.randn(), name="W")
    b = tf.Variable(np.random.randn(), name="b")

    pred = tf.add(tf.multiply(x, w), b)
    pred = tf.identity(pred, name="pred")
    cost = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * n_samples)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Fit all training data
        for _ in range(training_epochs):
            for (ix, iy) in zip(_train_x, _train_y):
                sess.run(optimizer, feed_dict={x: ix, y: iy})
        training_cost = sess.run(cost, feed_dict={x: _train_x, y: _train_y})
        testing_cost = sess.run(cost, feed_dict={x: _test_x, y: _test_y})
        print("train_cost={}, test_cost={}, diff={}"
              .format(training_cost, testing_cost, abs(training_cost - testing_cost)))

        p = os.path.abspath(os.path.join(model_path, "checkpoint"))
        os.makedirs(p, exist_ok=True)
        p = saver.save(sess, os.path.join(p, "model"))

        frozen_graph = freeze_session(sess, output_names=["pred:0"])
        p = os.path.abspath(os.path.join(model_path, "graphdef"))
        tf.train.write_graph(frozen_graph, p, "frozen.pb", as_text=False)

        p = os.path.abspath(os.path.join(model_path, "saved_model"))
        tf.saved_model.simple_save(sess, p, inputs={"X": x}, outputs={"pred": pred})


train("models/regression")
