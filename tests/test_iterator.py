#!/usr/bin/env python
# coding=utf-8

"""
@usage:
-> tensorflow {low level api} to onnx compatibility test
-> user should define model they wish to test in define_rnn()

@authors: [950630] at [10/09/2019]

-> max.dillon@pwc.com
"""

import os
import sys

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import onnx

import tensorflow.compat.v1 as tf
import tf2onnx
from tf2onnx import loader

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


class BatchIteratorConversionTest:
    """  """

    def __init__(self, batch_sz=64):
        self.X, self.y = make_blobs(n_samples=250, centers=3, n_features=5, random_state=0)

        self.batch_sz = batch_sz
        self.learning_rate = 0.01
        self.num_epochs = 10

        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X, self.y,
                                                                                        test_size=0.3, random_state=7)

    def create_tf_datasets(self):
        """"""
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.batch(self.batch_sz)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.next_element = iterator.get_next()

        self.train_init_op = iterator.make_initializer(train_dataset)

    def variables_placeholders(self):
        """ """
        self.batchX_placeholder = tf.placeholder(tf.float32, shape=[None, self.X_train.shape[1]], name="input")
        self.batchy_placeholder = tf.placeholder(tf.float32, shape=[None, self.y_train.shape[1]])

    def define_model(self):
        """ """

        linear_layer = tf.layers.Dense(units=self.y_train.shape[1], activation='softmax')
        y_pred = linear_layer(self.next_element)
        y_pred = tf.identity(y_pred, name="output")

        return y_pred

    def define_metrics(self):
        """ """

        y_pred = self.define_model()

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.next_element[1],
                                                    logits=y_pred)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train = optimizer.minimize(self.loss)

    def train_model(self):
        """ trains model and outputs a frozen graph which can be saved """

        # need to initialise all variables to use them in sess
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(self.train_init_op)

            for i in range(self.num_epochs):
                _, loss_out = sess.run((self.train, self.loss))

                print(loss_out)

            self.output_graph_def = loader.freeze_session(sess, output_names=["output:0"])

    def test_convert_onnx(self):
        """ convert model into onnx and save it as 'model_tensorflow.onnx' """

        tf.reset_default_graph()
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(self.output_graph_def, name='')

            self.onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names=["input:0"],
                                                         output_names=["output:0"])
        
        assert type(self.onnx_graph) == 'tf2onnx.graph.Graph'


if __name__ == '__main__':

    print("\n -- Testing onnx compatability with TensorFlow -- ", flush=True)

    print(f"\n -- Testing tensorflow version {tf.__version__} -- ", flush=True)

    print(f"\n -- Testing onnx version {onnx.__version__} -- ", flush=True)

    print(f"\n -- Testing tf2onnx version {tf2onnx.__version__} -- ", flush=True)

    iterator_test = BatchIteratorConversionTest()
    iterator_test.create_tf_datasets()

    iterator_test.variables_placeholders()
    iterator_test.define_metrics()
    iterator_test.train_model()
    iterator_test.test_convert_onnx()

    print("\n -- TensorFlow model successfully saved as model.onnx -- \n", flush=True)
