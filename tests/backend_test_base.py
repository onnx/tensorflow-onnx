# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Test Base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,import-outside-toplevel
# pylint: disable=wrong-import-position

import logging
import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import lookup_ops
from common import get_test_config
from tf2onnx import utils
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import optimizer
from tf2onnx.tf_loader import tf_reset_default_graph, tf_session, tf_placeholder, from_function, freeze_session
from tf2onnx.tf_loader import tf_optimize, is_tf2, get_hash_table_info
from tf2onnx.tf_utils import compress_graph_def
from tf2onnx.graph import ExternalTensorStorage


if is_tf2():
    tf_set_random_seed = tf.compat.v1.set_random_seed
    tf_tables_initializer = tf.compat.v1.tables_initializer
    tf_lite = tf.compat.v1.lite
else:
    tf_set_random_seed = tf.set_random_seed
    tf_tables_initializer = tf.tables_initializer
    tf_lite = None


class Tf2OnnxBackendTestBase(unittest.TestCase):
    def setUp(self):
        self.config = get_test_config()
        tf_reset_default_graph()
        # reset name generation on every test
        utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.
        self.logger = logging.getLogger(self.__class__.__name__)

    def tearDown(self):
        if not self.config.is_debug_mode:
            utils.delete_directory(self.test_data_directory)

    @property
    def test_data_directory(self):
        return os.path.join(self.config.temp_dir, self._testMethodName)

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

    def run_onnxruntime(self, model_path, inputs, output_names):
        """Run test against onnxruntime backend."""
        import onnxruntime as rt
        opt = rt.SessionOptions()
        # in case of issues with the runtime, one can enable more logging
        # opt.log_severity_level = 0
        # opt.log_verbosity_level = 255
        # opt.enable_profiling = True
        m = rt.InferenceSession(model_path, opt)
        results = m.run(output_names, inputs)
        return results

    def run_backend(self, g, outputs, input_dict, large_model=False, postfix=""):
        tensor_storage = ExternalTensorStorage() if large_model else None
        model_proto = g.make_model("test", external_tensor_storage=tensor_storage)
        model_path = self.save_onnx_model(model_proto, input_dict, external_tensor_storage=tensor_storage, postfix=postfix)

        if self.config.backend == "onnxruntime":
            y = self.run_onnxruntime(model_path, input_dict, outputs)
        elif self.config.backend == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def run_test_case(self, func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-07, atol=1e-5,
                      convert_var_to_const=True, constant_fold=True, check_value=True, check_shape=True,
                      check_dtype=True, process_args=None, onnx_feed_dict=None, graph_validator=None, as_session=False,
                      large_model=False, test_tflite=True, skip_tfl_consistency_check=False, premade_placeholders=False):
        # optional - passed to process_tf_graph
        if process_args is None:
            process_args = {}
        # optional - pass distinct feed_dict to onnx runtime
        if onnx_feed_dict is None:
            onnx_feed_dict = feed_dict
        input_names_with_port = list(feed_dict)
        tf_reset_default_graph()
        graph_def = None
        initialized_tables = None
        if tf_lite is None:
            test_tflite = False

        np.random.seed(1)  # Make it reproducible.
        clean_feed_dict = {utils.node_name(k): v for k, v in feed_dict.items()}
        if is_tf2() and not as_session:
            #
            # use eager to execute the tensorflow func
            #
            # numpy doesn't work for all ops, make it tf.Tensor()
            input_tensors = [tf.TensorSpec(shape=v.shape, dtype=tf.as_dtype(v.dtype), name=utils.node_name(k))
                             for k, v in feed_dict.items()]
            input_list = [tf.convert_to_tensor(v, dtype=tf.as_dtype(v.dtype), name=utils.node_name(k))
                          for k, v in feed_dict.items()]
            tf.random.set_seed(1)
            expected = func(*input_list)
            if isinstance(expected, (list, tuple)):
                # list or tuple
                expected = [x.numpy() for x in expected]
            else:
                # single result
                expected = [expected.numpy()]

            # now make the eager functions a graph
            concrete_func = tf.function(func, input_signature=tuple(input_tensors))
            concrete_func = concrete_func.get_concrete_function()
            graph_def = from_function(concrete_func,
                                      input_names=list(feed_dict.keys()),
                                      output_names=output_names_with_port,
                                      large_model=large_model)
        else:
            #
            # use graph to execute the tensorflow func
            #
            with tf_session() as sess:
                tf_set_random_seed(1)
                input_list = []
                if not premade_placeholders:
                    for k, v in clean_feed_dict.items():
                        input_list.append(tf_placeholder(name=k, shape=v.shape, dtype=tf.as_dtype(v.dtype)))
                func(*input_list)
                variables_lib.global_variables_initializer().run()
                tf_tables_initializer().run()

                output_dict = []
                for out_name in output_names_with_port:
                    output_dict.append(sess.graph.get_tensor_by_name(out_name))
                expected = sess.run(output_dict, feed_dict=feed_dict)
                graph_def = freeze_session(sess,
                                           input_names=list(feed_dict.keys()),
                                           output_names=output_names_with_port)
                table_names, key_dtypes, value_dtypes = get_hash_table_info(graph_def)
                initialized_tables = {}
                for n, k_dtype, val_dtype in zip(table_names, key_dtypes, value_dtypes):
                    h = lookup_ops.hash_table_v2(k_dtype, val_dtype, shared_name=n)
                    k, v = lookup_ops.lookup_table_export_v2(h, k_dtype, val_dtype)
                    initialized_tables[n] = (sess.run(k), sess.run(v))

            tf_reset_default_graph()
            with tf_session() as sess:
                tf.import_graph_def(graph_def, name='')
                graph_def = tf_optimize(list(feed_dict.keys()), output_names_with_port,
                                        graph_def, fold_constant=constant_fold)

        tf_reset_default_graph()
        with tf_session() as sess:
            const_node_values = None
            if large_model:
                const_node_values = compress_graph_def(graph_def)
            tf.import_graph_def(graph_def, name='')

            if not feed_dict:
                test_tflite = False  # Can't make TFlite model with no inputs

            if test_tflite:
                sess_inputs = [sess.graph.get_tensor_by_name(k) for k in feed_dict.keys()]
                sess_outputs = [sess.graph.get_tensor_by_name(n) for n in output_names_with_port]
                converter = tf_lite.TFLiteConverter.from_session(sess, sess_inputs, sess_outputs)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                #converter.inference_input_type = tf.int8  # or tf.uint8
                #converter.inference_output_type = tf.int8  # or tf.uint8
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                # def representative_dataset_gen():
                #     yield [np.array([[4, 3, 1, 0],
                #           [2, 1, 0, 1],
                #           [1, 2, 4, 1],
                #           [3, 1, 0, 2]], dtype=np.float32).reshape([1, 4, 4, 1])]
                # converter.representative_dataset = representative_dataset_gen

                from tensorflow.lite.python.convert import ConverterError
                try:
                    tflite_model = converter.convert()
                    tflite_path = os.path.join(self.test_data_directory, self._testMethodName + ".tflite")
                    dir_name = os.path.dirname(tflite_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    with open(tflite_path, 'wb') as f:
                        f.write(tflite_model)
                except ConverterError as e:
                    test_tflite = False


            if True or self.config.is_debug_mode:
                model_path = os.path.join(self.test_data_directory, self._testMethodName + "_after_tf_optimize.pb")
                utils.save_protobuf(model_path, graph_def)
                self.logger.debug("created file  %s", model_path)

            g = process_tf_graph(sess.graph, opset=self.config.opset,
                                 input_names=list(feed_dict.keys()),
                                 output_names=output_names_with_port,
                                 target=self.config.target,
                                 const_node_values=const_node_values,
                                 initialized_tables=initialized_tables,
                                 **process_args)
            g = optimizer.optimize_graph(g, catch_errors=False)
            actual = self.run_backend(g, output_names_with_port, onnx_feed_dict, large_model)

        for expected_val, actual_val in zip(expected, actual):
            if check_value:
                if expected_val.dtype == np.object:
                    decode = np.vectorize(lambda x: x.decode('UTF-8'))
                    expected_val_str = decode(expected_val)
                    self.assertAllEqual(expected_val_str, actual_val)
                else:
                    self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=atol)
            if check_dtype:
                self.assertEqual(expected_val.dtype, actual_val.dtype)
            # why need shape checke: issue when compare [] with scalar
            # https://github.com/numpy/numpy/issues/11071
            if check_shape:
                self.assertEqual(expected_val.shape, actual_val.shape)

        if graph_validator:
            self.assertTrue(graph_validator(g))

        if test_tflite:
            #feed_dict['input:0'] = np.array([10, 20, 100, 200], dtype=np.int8).reshape((2, 2))
            try:
                # tflite is a hot mess so sometimes is converts from tf but produces an invalid model
                interpreter = tf.lite.Interpreter(tflite_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                input_name_to_index = {n['name'].split(':')[0]: n['index'] for n in input_details}
                ouput_name_to_index = {n['name'].split(':')[0]: n['index'] for n in output_details}
                feed_dict_without_port = {k.split(':')[0]: v for k, v in feed_dict.items()}
                # The output names might be different in the tflite but the order is the same
                output_names = [n['name'] for n in output_details]
                for k, v in feed_dict_without_port.items():
                    interpreter.set_tensor(input_name_to_index[k], v)
                interpreter.invoke()
                tf_lite_output_data = [interpreter.get_tensor(output['index']) for output in output_details]
            except (RuntimeError, ValueError) as e:
                test_tflite = False
        if test_tflite:
            if not skip_tfl_consistency_check:
                for expected_val, tf_lite_val in zip(expected, tf_lite_output_data):
                    if check_value:
                        self.assertAllClose(expected_val, tf_lite_val, rtol=rtol, atol=atol)
                    if check_dtype:
                        self.assertEqual(expected_val.dtype, tf_lite_val.dtype)
                    # why need shape checke: issue when compare [] with scalar
                    # https://github.com/numpy/numpy/issues/11071
                    if check_shape:
                        self.assertEqual(expected_val.shape, tf_lite_val.shape)

            tfl_process_args = process_args.copy()
            if 'inputs_as_nchw' in tfl_process_args:
                inp_with_port = tfl_process_args['inputs_as_nchw']
                tfl_process_args['inputs_as_nchw'] = [i.split(':')[0] for i in inp_with_port]

            g = process_tf_graph(None, opset=self.config.opset,
                                 input_names=list(feed_dict_without_port.keys()),
                                 output_names=output_names,
                                 target=self.config.target,
                                 const_node_values=const_node_values,
                                 tflite_path=tflite_path,
                                 **tfl_process_args)
            g = optimizer.optimize_graph(g)
            onnx_feed_dict_without_port = {k.split(':')[0]: v for k, v in onnx_feed_dict.items()}
            onnx_from_tfl_output = self.run_backend(g, output_names, onnx_feed_dict_without_port, postfix="_from_tflite")


            for tf_lite_val, onnx_val in zip(tf_lite_output_data, onnx_from_tfl_output):
                if check_value:
                    self.assertAllClose(tf_lite_val, onnx_val, rtol=rtol, atol=atol)
                if check_dtype:
                    self.assertEqual(tf_lite_val.dtype, onnx_val.dtype)
                # why need shape checke: issue when compare [] with scalar
                # https://github.com/numpy/numpy/issues/11071
                if check_shape:
                    self.assertEqual(tf_lite_val.shape, onnx_val.shape)

            if graph_validator:
                self.assertTrue(graph_validator(g))

        return g

    def save_onnx_model(self, model_proto, feed_dict, postfix="", external_tensor_storage=None):
        target_path = utils.save_onnx_model(self.test_data_directory, self._testMethodName + postfix, feed_dict,
                                            model_proto, include_test_data=self.config.is_debug_mode,
                                            as_text=self.config.is_debug_mode,
                                            external_tensor_storage=external_tensor_storage)

        self.logger.debug("create model file: %s", target_path)
        return target_path
