# SPDX-License-Identifier: Apache-2.0


"""Unit Test Base."""

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,import-outside-toplevel
# pylint: disable=wrong-import-position,invalid-unary-operand-type

import logging
import os
import unittest
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import lookup_ops
import onnx
from common import get_test_config
from tfjs_runner import run_tfjs
from tf2onnx import constants
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

    def run_onnxruntime(self, model_path, inputs, output_names, use_custom_ops=False):
        """Run test against onnxruntime backend."""
        import onnxruntime as rt
        providers = ['CPUExecutionProvider']
        if rt.get_device() == "GPU":
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus is None or len(gpus) > 1:
                providers = ['CUDAExecutionProvider']
        opt = rt.SessionOptions()
        if use_custom_ops:
            from onnxruntime_extensions import get_library_path
            opt.register_custom_ops_library(get_library_path())

        # in case of issues with the runtime, one can enable more logging
        # opt.log_severity_level = 0
        # opt.log_verbosity_level = 255
        # opt.enable_profiling = True

        m = rt.InferenceSession(model_path, opt, providers=providers)
        results = m.run(output_names, inputs)
        return results

    def run_backend(self, g, outputs, input_dict, large_model=False, postfix="", use_custom_ops=False):
        tensor_storage = ExternalTensorStorage() if large_model else None
        model_proto = g.make_model("test", external_tensor_storage=tensor_storage)
        model_path = self.save_onnx_model(model_proto, input_dict, external_tensor_storage=tensor_storage,
                                          postfix=postfix)

        if self.config.backend == "onnxruntime":
            y = self.run_onnxruntime(model_path, input_dict, outputs, use_custom_ops)
        elif self.config.backend == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def assert_results_equal(self, expected, actual, rtol, atol, mtol=None,
                             check_value=True, check_shape=True, check_dtype=True):
        for expected_val, actual_val in zip(expected, actual):
            if check_value:
                if expected_val.dtype == object:
                    # TFLite pads strings with nul bytes
                    decode = np.vectorize(lambda x: x.replace(b'\x00', b'').decode('UTF-8'))
                    expected_val_str = decode(expected_val)
                    self.assertAllEqual(expected_val_str, actual_val)
                elif expected_val.dtype.kind == 'U':
                    self.assertAllEqual(expected_val, actual_val)
                else:
                    if mtol is not None:
                        expected_val = np.minimum(expected_val, mtol)
                        expected_val = np.maximum(expected_val, -mtol)
                        actual_val = np.minimum(actual_val, mtol)
                        actual_val = np.maximum(actual_val, -mtol)
                    self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=atol)
            if check_dtype:
                self.assertEqual(expected_val.dtype, actual_val.dtype)
            # why need shape checke: issue when compare [] with scalar
            # https://github.com/numpy/numpy/issues/11071
            if check_shape:
                self.assertEqual(expected_val.shape, actual_val.shape)

    def freeze_and_run_tf(self, func, feed_dict, outputs, as_session, premade_placeholders, large_model):
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
                table_info = get_hash_table_info(graph_def)
                initialized_tables = {}
                for info in table_info:
                    if info.shared_name is None:
                        continue
                    h = lookup_ops.hash_table_v2(info.key_dtype, info.val_dtype, shared_name=info.shared_name)
                    k, v = lookup_ops.lookup_table_export_v2(h, info.key_dtype, info.val_dtype)
                    initialized_tables[info.shared_name] = (sess.run(k), sess.run(v))

            tf_reset_default_graph()
            with tf_session() as sess:
                tf.import_graph_def(graph_def, name='')
                graph_def = tf_optimize(list(feed_dict.keys()), outputs, graph_def)

        return result, graph_def, initialized_tables

    def convert_to_tfjs(self, graph_def_path, output_names):
        try:
            from tensorflowjs.converters import converter
        except ImportError:
            self.logger.warning("Tensorflowjs.converters package imports failed.")
            return None
        tfjs_path = os.path.join(self.test_data_directory, self._testMethodName + "_tfjs")
        try:
            converter.convert([graph_def_path, tfjs_path, '--input_format', 'tf_frozen_model',
                               '--output_node_names', ','.join(output_names)])
        except ValueError:
            self.logger.warning("Convert tensorflowjs graph failed.")
            return None
        model_path = os.path.join(tfjs_path, 'model.json')
        if not os.path.exists(model_path):
            self.logger.warning("Tensorflowjs model path %s is empty.", model_path)
            return None
        return model_path

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
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,    # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS,      # enable TensorFlow flex ops.
            ]

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

    def tflite_has_supported_types(self, tflite_path):
        try:
            interpreter = tf.lite.Interpreter(tflite_path)
            tensor_details = interpreter.get_tensor_details()
            for tensor_detail in tensor_details:
                dtype = tensor_detail.get('dtype')
                if np.dtype(dtype).kind == 'O':
                    return False
            return True
        except (RuntimeError, ValueError):
            return False

    def run_tflite(self, tflite_path, feed_dict):
        try:
            interpreter = tf.lite.Interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_name_to_index = {n['name'].split(':')[0]: n['index'] for n in input_details}
            feed_dict_without_port = {k.split(':')[0]: v for k, v in feed_dict.items()}
            for k, v in feed_dict_without_port.items():
                interpreter.resize_tensor_input(input_name_to_index[k], v.shape)
            interpreter.allocate_tensors()
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

    def assert_shapes_correct(self, graph, allow_missing=False, run_checker=True, check_shape=True):
        if not check_shape:
            return None

        model_proto = graph.make_model("test")

        if run_checker and not any(graph.get_shape(out) is None for out in graph.outputs + graph.input_names):
            try:
                onnx.checker.check_model(model_proto, full_check=True)
            except onnx.shape_inference.InferenceError as e:
                # onnx checker verifies number of subgraph inputs incorrectly in IR 3
                if re.search(r"Graph has \d* inputs but \d* were provided", str(e)):
                    run_checker = False
                else:
                    raise e

        model_shapes = onnx.shape_inference.infer_shapes(model_proto)
        def get_shape(info):
            if not info.type.tensor_type.HasField("shape"):
                return None
            return [d.dim_value if d.HasField('dim_value') else -1 for d in info.type.tensor_type.shape.dim]
        def get_dtype(info):
            tensor_type = info.type.tensor_type
            is_seq = False
            result = None
            if info.type.HasField("sequence_type"):
                tensor_type = info.type.sequence_type.elem_type.tensor_type
                is_seq = True
            if tensor_type.HasField("elem_type"):
                result = tensor_type.elem_type
            return utils.SeqType(result) if is_seq else result
        for info in model_shapes.graph.value_info:
            if info.name == "":
                continue
            onnx_shape = get_shape(info)
            tf2onnx_shape = graph.get_shape(info.name)
            if onnx_shape is None:
                continue
            if allow_missing and tf2onnx_shape is None:
                continue
            self.assertTrue(tf2onnx_shape is not None)
            if -1 in onnx_shape or (allow_missing and -1 in tf2onnx_shape):
                self.assertEqual(len(onnx_shape), len(tf2onnx_shape))
                for d1, d2 in zip(onnx_shape, tf2onnx_shape):
                    if d1 != -1 and (d2 != -1 or not allow_missing):
                        self.assertEqual(d1, d2)
            else:
                self.assertEqual(onnx_shape, tf2onnx_shape)
            self.assertEqual(get_dtype(info), graph.get_dtype(info.name))

    def run_test_case(self, func, feed_dict, input_names_with_port, output_names_with_port,
                      rtol=1e-07, atol=1e-5, mtol=None, convert_var_to_const=True, check_value=True,
                      check_shape=True, check_dtype=True, process_args=None, onnx_feed_dict=None,
                      graph_validator=None, as_session=False, large_model=False, premade_placeholders=False,
                      use_custom_ops=False, optimize=True):
        """
        This function tests all scenarios available through the command line.
        The command line always runs the optimizers.
        However, they may modify the final graph into something different than the
        tested converter implements. Set `optimize=False` to keep the original
        set of nodes and helps debugging. However, the same function should
        be called with `optimize=True` to test what the user would actually get.
        """
        test_tf = not self.config.skip_tf_tests
        test_tflite = not self.config.skip_tflite_tests
        test_tfjs = not self.config.skip_tfjs_tests
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
                                   premade_placeholders, large_model)

        graph_def_path = os.path.join(self.test_data_directory, self._testMethodName + "_after_tf_optimize.pb")
        utils.save_protobuf(graph_def_path, graph_def)
        self.logger.debug("created file  %s", graph_def_path)
        tfl_process_args = process_args.copy()

        if test_tfjs:
            tfjs_path = self.convert_to_tfjs(graph_def_path, output_names_with_port)
            if tfjs_path is None:
                test_tfjs = False

        if test_tflite:
            tflite_path = self.convert_to_tflite(graph_def, feed_dict, output_names_with_port)
            test_tflite = tflite_path is not None and self.tflite_has_supported_types(tflite_path)

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
                if optimize:
                    g = optimizer.optimize_graph(g, catch_errors=False)
                actual = self.run_backend(g, output_names_with_port, onnx_feed_dict, large_model,
                                          use_custom_ops=use_custom_ops)
            if 'outputs_as_nchw' in tfl_process_args:
                for output_name in tfl_process_args['outputs_as_nchw']:
                    i = output_names_with_port.index(output_name)
                    actual[i] = np.transpose(actual[i], constants.NCHW_TO_NHWC)

            self.assert_results_equal(expected, actual, rtol, atol, mtol, check_value, check_shape,
                                      check_dtype)
            self.assert_shapes_correct(g, self.config.allow_missing_shapes, not self.config.skip_onnx_checker,
                                       check_shape)

            if graph_validator:
                self.assertTrue(graph_validator(g))

        if test_tflite:
            tfl_res, tfl_outputs = self.run_tflite(tflite_path, feed_dict)
            test_tflite = tfl_res is not None

        if test_tflite:
            if run_tfl_consistency_test:
                self.assert_results_equal(expected, tfl_res, rtol, atol, mtol, check_value, check_shape, check_dtype)

            if 'inputs_as_nchw' in tfl_process_args:
                nchw_inps_with_port = tfl_process_args['inputs_as_nchw']
                tfl_process_args['inputs_as_nchw'] = [i.split(':')[0] for i in nchw_inps_with_port]
            input_names_without_port = [inp.split(':')[0] for inp in feed_dict.keys()]
            if 'outputs_as_nchw' in tfl_process_args:
                nchw_outps_with_port = tfl_process_args['outputs_as_nchw']
                tfl_process_args['outputs_as_nchw'] = [i.split(':')[0] for i in nchw_outps_with_port]
                output_names_with_port = [i.split(':')[0] for i in nchw_outps_with_port]
            g = process_tf_graph(None, opset=self.config.opset,
                                 input_names=input_names_without_port,
                                 output_names=tfl_outputs,
                                 target=self.config.target,
                                 tflite_path=tflite_path,
                                 **tfl_process_args)
            if optimize:
                g = optimizer.optimize_graph(g)
            onnx_feed_dict_without_port = {k.split(':')[0]: v for k, v in onnx_feed_dict.items()}
            onnx_tfl_res = self.run_backend(g, tfl_outputs, onnx_feed_dict_without_port,
                                            postfix="_from_tflite", use_custom_ops=use_custom_ops)
            if 'outputs_as_nchw' in tfl_process_args:
                for output_name in tfl_process_args['outputs_as_nchw']:
                    i = output_names_with_port.index(output_name)
                    onnx_tfl_res[i] = np.transpose(onnx_tfl_res[i], constants.NCHW_TO_NHWC)

            self.assert_results_equal(tfl_res, onnx_tfl_res, rtol, atol, mtol, check_value, check_shape, check_dtype)
            self.assert_shapes_correct(g, self.config.allow_missing_shapes, not self.config.skip_onnx_checker,
                                       check_shape)

            if graph_validator:
                self.assertTrue(graph_validator(g))

        if test_tfjs:
            try:
                tfjs_res = run_tfjs(tfjs_path, feed_dict)
            except RuntimeError as e:
                ignored_errors = ["is not yet supported", "Operands could not be broadcast together",
                                  "unknown dtype null", "must be [NaN", "Cannot read property 'name' of undefined",
                                  "Either strides or dilations must be 1", "does not support"]
                if any(err in str(e) for err in ignored_errors):
                    test_tfjs = False
                else:
                    raise e

        if test_tfjs:
            g = process_tf_graph(None, opset=self.config.opset,
                                 input_names=list(feed_dict.keys()),
                                 output_names=None,
                                 target=self.config.target,
                                 tfjs_path=tfjs_path,
                                 **process_args)
            g = optimizer.optimize_graph(g)
            onnx_tfjs_res = self.run_backend(g, None, onnx_feed_dict, large_model,
                                             postfix="_from_tfjs", use_custom_ops=use_custom_ops)
            if 'outputs_as_nchw' in tfl_process_args:
                for output_name in tfl_process_args['outputs_as_nchw']:
                    i = output_names_with_port.index(output_name)
                    onnx_tfjs_res[i] = np.transpose(onnx_tfjs_res[i], constants.NCHW_TO_NHWC)

            self.assert_results_equal(tfjs_res, onnx_tfjs_res, rtol, atol, mtol, check_value, check_shape,
                                      check_dtype=False)
            self.assert_shapes_correct(g, self.config.allow_missing_shapes, not self.config.skip_onnx_checker,
                                       check_shape)

            if graph_validator:
                self.assertTrue(graph_validator(g))


        if g is None:
            raise unittest.SkipTest("tf, tflite, and tfjs marked to skip")
        return g

    def save_onnx_model(self, model_proto, feed_dict, postfix="", external_tensor_storage=None):
        target_path = utils.save_onnx_model(self.test_data_directory, self._testMethodName + postfix, feed_dict,
                                            model_proto, include_test_data=self.config.is_debug_mode,
                                            as_text=self.config.is_debug_mode,
                                            external_tensor_storage=external_tensor_storage)

        self.logger.debug("create model file: %s", target_path)
        return target_path
