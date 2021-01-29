# SPDX-License-Identifier: Apache-2.0


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
        model_path = self.save_onnx_model(model_proto, input_dict, external_tensor_storage=tensor_storage,
                                          postfix=postfix)

        if self.config.backend == "onnxruntime":
            y = self.run_onnxruntime(model_path, input_dict, outputs)
        elif self.config.backend == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def assert_results_equal(self, expected, actual, rtol, atol, check_value=True, check_shape=True, check_dtype=True):
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

    def freeze_and_run_tf(self, func, feed_dict, outputs, as_session, premade_placeholders, large_model, constant_fold):
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
            result = func(*input_list)
            if isinstance(result, (list, tuple)):
                # list or tuple
                result = [x.numpy() for x in result]
            else:
                # single result
                result = [result.numpy()]

            # now make the eager functions a graph
            concrete_func = tf.function(func, input_signature=tuple(input_tensors))
            concrete_func = concrete_func.get_concrete_function()
            graph_def = from_function(concrete_func,
                                      input_names=list(feed_dict.keys()),
                                      output_names=outputs,
                                      large_model=large_model)
            initialized_tables = None
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
                for out_name in outputs:
                    output_dict.append(sess.graph.get_tensor_by_name(out_name))
                result = sess.run(output_dict, feed_dict=feed_dict)
                graph_def = freeze_session(sess,
                                           input_names=list(feed_dict.keys()),
                                           output_names=outputs)
                table_names, key_dtypes, value_dtypes = get_hash_table_info(graph_def)
                initialized_tables = {}
                for n, k_dtype, val_dtype in zip(table_names, key_dtypes, value_dtypes):
                    h = lookup_ops.hash_table_v2(k_dtype, val_dtype, shared_name=n)
                    k, v = lookup_ops.lookup_table_export_v2(h, k_dtype, val_dtype)
                    initialized_tables[n] = (sess.run(k), sess.run(v))

            tf_reset_default_graph()
            with tf_session() as sess:
                tf.import_graph_def(graph_def, name='')
                graph_def = tf_optimize(list(feed_dict.keys()), outputs, graph_def, fold_constant=constant_fold)

        if True or self.config.is_debug_mode:
            model_path = os.path.join(self.test_data_directory, self._testMethodName + "_after_tf_optimize.pb")
            utils.save_protobuf(model_path, graph_def)
            self.logger.debug("created file  %s", model_path)
        return result, graph_def, initialized_tables

    def convert_to_tflite(self, graph_def, feed_dict, outputs):
        if not feed_dict:
            return None   # Can't make TFlite model with no inputs
        tf_reset_default_graph()
        with tf_session() as sess:
            tf.import_graph_def(graph_def, name='')
            sess_inputs = [sess.graph.get_tensor_by_name(k) for k in feed_dict.keys()]
            sess_outputs = [sess.graph.get_tensor_by_name(n) for n in outputs]
            converter = tf_lite.TFLiteConverter.from_session(sess, sess_inputs, sess_outputs)
            #converter.optimizations = [tf.lite.Optimize.DEFAULT]

            from tensorflow.lite.python.convert import ConverterError
            try:
                tflite_model = converter.convert()
                tflite_path = os.path.join(self.test_data_directory, self._testMethodName + ".tflite")
                dir_name = os.path.dirname(tflite_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                return tflite_path
            except ConverterError:
                return None

    def run_tflite(self, tflite_path, feed_dict):
        try:
            interpreter = tf.lite.Interpreter(tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_name_to_index = {n['name'].split(':')[0]: n['index'] for n in input_details}
            feed_dict_without_port = {k.split(':')[0]: v for k, v in feed_dict.items()}
            # The output names might be different in the tflite but the order is the same
            output_names = [n['name'] for n in output_details]
            for k, v in feed_dict_without_port.items():
                interpreter.set_tensor(input_name_to_index[k], v)
            interpreter.invoke()
            result = [interpreter.get_tensor(output['index']) for output in output_details]
            return result, output_names
        except (RuntimeError, ValueError):
            # tflite sometimes converts from tf but produces an invalid model
            return None, None

    def run_test_case(self, func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-07, atol=1e-5,
                      convert_var_to_const=True, constant_fold=True, check_value=True, check_shape=True,
                      check_dtype=True, process_args=None, onnx_feed_dict=None, graph_validator=None, as_session=False,
                      large_model=False, premade_placeholders=False):
        test_tf = not self.config.skip_tf_tests
        test_tflite = not self.config.skip_tflite_tests
        run_tfl_consistency_test = test_tf and test_tflite and self.config.run_tfl_consistency_test
        # optional - passed to process_tf_graph
        if process_args is None:
            process_args = {}
        # optional - pass distinct feed_dict to onnx runtime
        if onnx_feed_dict is None:
            onnx_feed_dict = feed_dict
        input_names_with_port = list(feed_dict)
        tf_reset_default_graph()
        if tf_lite is None:
            test_tflite = False
        g = None

        expected, graph_def, initialized_tables = \
            self.freeze_and_run_tf(func, feed_dict, output_names_with_port, as_session,
                                   premade_placeholders, large_model, constant_fold)

        if test_tflite:
            tflite_path = self.convert_to_tflite(graph_def, feed_dict, output_names_with_port)
            test_tflite = tflite_path is not None

        if test_tf:
            tf_reset_default_graph()
            with tf_session() as sess:
                const_node_values = None
                if large_model:
                    const_node_values = compress_graph_def(graph_def)
                tf.import_graph_def(graph_def, name='')

                g = process_tf_graph(sess.graph, opset=self.config.opset,
                                     input_names=list(feed_dict.keys()),
                                     output_names=output_names_with_port,
                                     target=self.config.target,
                                     const_node_values=const_node_values,
                                     initialized_tables=initialized_tables,
                                     **process_args)
                g = optimizer.optimize_graph(g, catch_errors=False)
                actual = self.run_backend(g, output_names_with_port, onnx_feed_dict, large_model)

            self.assert_results_equal(expected, actual, rtol, atol, check_value, check_shape, check_dtype)

            if graph_validator:
                self.assertTrue(graph_validator(g))

        if test_tflite:
            tfl_results, tfl_outputs = self.run_tflite(tflite_path, feed_dict)
            test_tflite = tfl_results is not None

        if test_tflite:
            if run_tfl_consistency_test:
                self.assert_results_equal(expected, tfl_results, rtol, atol, check_value, check_shape, check_dtype)

            tfl_process_args = process_args.copy()
            if 'inputs_as_nchw' in tfl_process_args:
                nchw_inps_with_port = tfl_process_args['inputs_as_nchw']
                tfl_process_args['inputs_as_nchw'] = [i.split(':')[0] for i in nchw_inps_with_port]
            input_names_without_port = [inp.split(':')[0] for inp in feed_dict.keys()]

            g = process_tf_graph(None, opset=self.config.opset,
                                 input_names=input_names_without_port,
                                 output_names=tfl_outputs,
                                 target=self.config.target,
                                 tflite_path=tflite_path,
                                 **tfl_process_args)
            g = optimizer.optimize_graph(g)
            onnx_feed_dict_without_port = {k.split(':')[0]: v for k, v in onnx_feed_dict.items()}
            onnx_from_tfl_res = self.run_backend(g, tfl_outputs, onnx_feed_dict_without_port, postfix="_from_tflite")

            self.assert_results_equal(tfl_results, onnx_from_tfl_res, rtol, atol, check_value, check_shape, check_dtype)

            if graph_validator:
                self.assertTrue(graph_validator(g))

        if g is None:
            raise unittest.SkipTest("Both tf and tflite marked to skip")
        return g

    def save_onnx_model(self, model_proto, feed_dict, postfix="", external_tensor_storage=None):
        target_path = utils.save_onnx_model(self.test_data_directory, self._testMethodName + postfix, feed_dict,
                                            model_proto, include_test_data=self.config.is_debug_mode,
                                            as_text=self.config.is_debug_mode,
                                            external_tensor_storage=external_tensor_storage)

        self.logger.debug("create model file: %s", target_path)
        return target_path
