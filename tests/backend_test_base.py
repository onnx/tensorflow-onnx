# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Test Base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables as variables_lib
from tf2onnx import utils
from tf2onnx.tfonnx import process_tf_graph, tf_optimize, DEFAULT_TARGET, POSSIBLE_TARGETS


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class Tf2OnnxBackendTestBase(unittest.TestCase):
    # static variables
    TMPPATH = tempfile.mkdtemp()
    BACKEND = os.environ.get("TF2ONNX_TEST_BACKEND", "onnxruntime")
    OPSET = int(os.environ.get("TF2ONNX_TEST_OPSET", 7))
    TARGET = os.environ.get("TF2ONNX_TEST_TARGET", "").split(",")
    DEBUG = None

    def debug_mode(self):
        return type(self).DEBUG

    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        utils.INTERNAL_NAME = 1
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
        return results

    def run_onnxmsrtnext(self, model_path, inputs, output_names):
        """Run test against msrt-next backend."""
        import lotus
        m = lotus.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    def run_onnxruntime(self, model_path, inputs, output_names):
        """Run test against msrt-next backend."""
        import onnxruntime as rt
        m = rt.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    def _run_backend(self, g, outputs, input_dict):
        model_proto = g.make_model("test")
        model_path = self.save_onnx_model(model_proto, input_dict)
        if type(self).BACKEND == "onnxmsrtnext":
            y = self.run_onnxmsrtnext(model_path, input_dict, outputs)
        elif type(self).BACKEND == "onnxruntime":
            y = self.run_onnxruntime(model_path, input_dict, outputs)
        elif type(self).BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def run_test_case(self, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-07,
                      convert_var_to_const=True, constant_fold=True, check_value=True, check_shape=False,
                      check_dtype=False, process_args=None, onnx_feed_dict=None):
        # optional - passed to process_tf_graph
        if process_args is None:
            process_args = {}
        # optional - pass distinct feed_dict to onnx runtime
        if onnx_feed_dict is None:
            onnx_feed_dict = feed_dict

        graph_def = None
        save_dir = os.path.join(type(self).TMPPATH, self._testMethodName)

        if convert_var_to_const:
            with tf.Session() as sess:
                variables_lib.global_variables_initializer().run()
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

        if self.debug_mode():
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, self._testMethodName + "_original.pb")
            with open(model_path, "wb") as f:
                f.write(sess.graph_def.SerializeToString())
            self.log.debug("created file %s", model_path)

        graph_def = tf_optimize(input_names_with_port, output_names_with_port,
                                sess.graph_def, constant_fold)

        if self.debug_mode() and constant_fold:
            model_path = os.path.join(save_dir, self._testMethodName + "_after_tf_optimize.pb")
            with open(model_path, "wb") as f:
                f.write(graph_def.SerializeToString())
            self.log.debug("created file  %s", model_path)

        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            g = process_tf_graph(sess.graph, opset=type(self).OPSET, output_names=output_names_with_port,
                                 target=type(self).TARGET, **process_args)
            actual = self._run_backend(g, output_names_with_port, onnx_feed_dict)

        for expected_val, actual_val in zip(expected, actual):
            if check_value:
                self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=0.)
            if check_dtype:
                self.assertEqual(expected_val.dtype, actual_val.dtype)
            if check_shape:
                self.assertEqual(expected_val.shape, actual_val.shape)

    def save_onnx_model(self, model_proto, feed_dict, postfix=""):
        save_path = os.path.join(type(self).TMPPATH, self._testMethodName)
        target_path = utils.save_onnx_model(save_path, self._testMethodName + postfix, feed_dict, model_proto,
                                            include_test_data=self.debug_mode(), as_text=self.debug_mode())

        self.log.debug("create model file: %s", target_path)
        return target_path

    @staticmethod
    def trigger(ut_class):
        parser = argparse.ArgumentParser()
        parser.add_argument('--backend', default=Tf2OnnxBackendTestBase.BACKEND,
                            choices=["caffe2", "onnxmsrtnext", "onnxruntime"],
                            help="backend to test against")
        parser.add_argument('--opset', type=int, default=Tf2OnnxBackendTestBase.OPSET, help="opset to test against")
        parser.add_argument("--target", default=",".join(DEFAULT_TARGET), choices=POSSIBLE_TARGETS,
                            help="target platform")
        parser.add_argument("--debug", help="output debugging information", action="store_true")
        parser.add_argument('unittest_args', nargs='*')

        args = parser.parse_args()
        print(args)
        ut_class.BACKEND = args.backend
        ut_class.OPSET = args.opset
        ut_class.DEBUG = args.debug
        ut_class.TARGET = args.target

        # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
        sys.argv[1:] = args.unittest_args
        unittest.main()
