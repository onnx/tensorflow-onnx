# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Test Base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables as variables_lib
import tf2onnx.utils
from tf2onnx.tfonnx import process_tf_graph

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class Tf2OnnxBackendTestBase(unittest.TestCase):
    # static variables
    TMPPATH = tempfile.mkdtemp()
    BACKEND = "onnxruntime"
    OPSET = 7
    DEBUG = None

    def debug_mode(self):
        return type(self).DEBUG

    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.

        self.log = logging.getLogger("tf2onnx.unitest." + str(type(self)))
        if self.debug_mode():
            self.log.setLevel(logging.DEBUG)
        else:
            # suppress log info of tensorflow so that result of test can be seen much easier
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.logging.set_verbosity(tf.logging.WARN)
            self.log.setLevel(logging.INFO)


    @staticmethod
    def assertAllClose(expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    @staticmethod
    def assertAllEqual(expected, actual, **kwargs):
        np.testing.assert_array_equal(expected, actual, **kwargs)

    def run_onnxcaffe2(self, onnx_graph, inputs):
        """Run test against caffe2 backend."""
        import caffe2.python.onnx.backend
        prepared_backend = caffe2.python.onnx.backend.prepare(onnx_graph)
        results = prepared_backend.run(inputs)
        return results[0]

    def run_onnxmsrtnext(self, onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(type(self).TMPPATH, test_name + ".onnx")
        self.log.debug("create model file: %s", model_path)
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

        m = lotus.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    def run_onnxruntime(self, onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import onnxruntime as rt
        model_path = os.path.join(type(self).TMPPATH, test_name + ".onnx")
        self.log.debug("create model file: %s", model_path)
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        m = rt.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    def _run_backend(self, g, outputs, input_dict):
        model_proto = g.make_model("test", outputs)
        if type(self).BACKEND == "onnxmsrtnext":
            y = self.run_onnxmsrtnext(model_proto, input_dict, outputs, self._testMethodName)
        elif type(self).BACKEND == "onnxruntime":
            y = self.run_onnxruntime(model_proto, input_dict, outputs, self._testMethodName)
        elif type(self).BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    # only when transform_tf_graph is true, input_names_with_port is necessary.
    def run_test_case(self, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-07,
                      convert_var_to_const=True, transform_tf_graph=True, check_value=True, check_shape=False,
                      check_dtype=False):
        graph_def = None
        if convert_var_to_const:
            with tf.Session() as sess:
                variables_lib.global_variables_initializer().run()
                #expected = sess.run(output_dict, feed_dict=feed_dict)
                output_name_without_port = [n.split(':')[0] for n in output_names_with_port]
                graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                         output_name_without_port)

            tf.reset_default_graph()
            tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            variables_lib.global_variables_initializer().run()
            output_dict = []
            for out_name in output_names_with_port:
                output_dict.append(sess.graph.get_tensor_by_name(out_name))

            expected = sess.run(output_dict, feed_dict=feed_dict)

        if transform_tf_graph:
            if self.debug_mode():
                model_path = os.path.join(type(self).TMPPATH, self._testMethodName + "_before_tf_optimize.pb")
                with open(model_path, "wb") as f:
                    f.write(sess.graph_def.SerializeToString())
                self.log.debug("created file %s", model_path)

            graph_def = tf2onnx.tfonnx.tf_optimize(input_names_with_port, output_names_with_port,
                                                   sess.graph_def, True)

            if self.debug_mode():
                model_path = os.path.join(type(self).TMPPATH, self._testMethodName + "_after_tf_optimize.pb")
                with open(model_path, "wb") as f:
                    f.write(graph_def.SerializeToString())
                self.log.debug("created file  %s", model_path)

            tf.reset_default_graph()
            tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            g = process_tf_graph(sess.graph, opset=type(self).OPSET) #continue_on_error=True,
            actual = self._run_backend(g, output_names_with_port, feed_dict)

        for expected_val, actual_val in zip(expected, actual):
            if check_value:
                self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=0.)
            if check_dtype:
                self.assertEqual(expected_val.dtype, actual_val.dtype)
            if check_shape:
                self.assertEqual(expected_val.shape, actual_val.shape)

    @staticmethod
    def trigger(ut_class):
        parser = argparse.ArgumentParser()
        parser.add_argument('--backend', default="onnxruntime",
                            choices=["caffe2", "onnxmsrtnext", "onnxruntime"],
                            help="backend to test against")

        parser.add_argument('--opset', type=int, default=7,
                            help="opset to test against")
        parser.add_argument("--debug", help="output debugging information", action="store_true")
        parser.add_argument('unittest_args', nargs='*')

        args = parser.parse_args()
        print(args)
        ut_class.BACKEND = args.backend
        ut_class.OPSET = args.opset
        ut_class.DEBUG = args.debug

        # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
        sys.argv[1:] = args.unittest_args
        unittest.main()
