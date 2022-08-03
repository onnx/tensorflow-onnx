# SPDX-License-Identifier: Apache-2.0


"""Tool to convert and test pre-trained tensorflow models."""

# pylint: disable=broad-except,logging-not-lazy,unused-argument,unnecessary-lambda,import-outside-toplevel
# pylint: disable=wrong-import-position,too-many-nested-blocks

import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
import random
from collections import namedtuple
from packaging.version import Version


import yaml
import numpy as np
import PIL.Image
import six

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# contrib ops are registered only when the module is imported, the following import statement is needed,
# otherwise tf runtime error will show up when the tf model is restored from pb file because of un-registered ops.
try:
    import tensorflow.contrib.rnn  # pylint: disable=unused-import
except:  # pylint: disable=bare-except
    # not needed for tf-2.0
    pass

try:
    import tensorflow_text  # pylint: disable=unused-import
except Exception as err:
    pass

from tf2onnx import tf_loader, logging, optimizer, utils, tf_utils, constants
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.tf_loader import tf_session, tf_reset_default_graph
from tf2onnx.graph import ExternalTensorStorage
from tfjs_runner import run_tfjs

logger = logging.getLogger("run_pretrained")

TEMP_DIR = os.path.join(utils.get_temp_directory(), "run_pretrained")
PERF_STEP = 10
PERF_TIME = 10

def get_img(shape, path, dtype, should_scale=True):
    """Get image as input."""
    resize_to = shape[1:3]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(dtype)
    img_np = np.stack([img_np] * shape[0], axis=0).reshape(shape)
    if should_scale:
        img_np = img_np / 255
    return img_np


def get_beach(shape):
    """Get beach image as input."""
    return get_img(shape, "beach.jpg", np.float32, should_scale=True)


def get_beach_uint8(shape):
    """Get beach image as uint8."""
    return get_img(shape, "ade20k.jpg", np.uint8, should_scale=False)


def get_car(shape):
    """Get car image as input."""
    return get_img(shape, "car.JPEG", np.float32, should_scale=True)


def get_ade20k(shape):
    """Get truck image from ade20k segmentation dataset."""
    return get_img(shape, "ade20k.jpg", np.float32, should_scale=True)


def get_ade20k_uint8(shape):
    """Get truck image from ade20k segmentation dataset."""
    return get_img(shape, "ade20k.jpg", np.uint8, should_scale=False)


def get_random(shape):
    """Get random input."""
    np.random.seed(42)
    return np.random.sample(shape).astype(np.float32)


def get_random256(shape):
    """Get random imput between 0 and 255."""
    np.random.seed(42)
    return np.round(np.random.sample(shape) * 256).astype(np.float32)


def get_ramp(shape):
    """Get ramp input."""
    size = np.prod(shape)
    return np.linspace(1, size, size).reshape(shape).astype(np.float32)


def get_ones(shape):
    """Get ones."""
    return np.ones(shape).astype(np.float32)

def get_zeros(shape):
    """Get zeros."""
    return np.zeros(shape).astype(np.float32)

def get_zeros_int32(shape):
    """Get zeros."""
    return np.zeros(shape).astype(np.int32)

def get_zeros_int64(shape):
    """Get zeros."""
    return np.zeros(shape).astype(np.int64)

def get_ones_int32(shape):
    """Get ones."""
    return np.ones(shape).astype(np.int32)

def get_small_rand_int32(shape):
    """Get random ints in range [1, 99]"""
    np.random.seed(42)
    return np.random.randint(low=1, high=100, size=shape, dtype=np.int32)

def get_zeros_then_ones(shape):
    """Fill half the tensor with zeros and the rest with ones"""
    cnt = np.prod(shape)
    zeros_cnt = cnt // 2
    ones_cnt = cnt - zeros_cnt
    return np.concatenate((np.zeros(zeros_cnt, dtype=np.int32), np.ones(ones_cnt, dtype=np.int32))).reshape(shape)

def get_wav(shape):
    """Get sound data."""
    return np.sin(np.linspace(-np.pi, np.pi, shape[0]), dtype=np.float32)

def get_sentences(shape):
    """Get sentences of shape"""
    words = "the quick brown fox jumps over a lazy dog".split(' ')
    random.seed(42)
    def get_sentence():
        length = random.randint(2, 7)
        return ' '.join(random.choice(words) for _ in range(length))
    return np.array([get_sentence() for _ in range(np.product(shape))]).reshape(shape)


_INPUT_FUNC_MAPPING = {
    "get_beach": get_beach,
    "get_beach_uint8": get_beach_uint8,
    "get_car": get_car,
    "get_ade20k": get_ade20k,
    "get_ade20k_uint8": get_ade20k_uint8,
    "get_random": get_random,
    "get_random256": get_random256,
    "get_ramp": get_ramp,
    "get_ones": get_ones,
    "get_zeros": get_zeros,
    "get_wav": get_wav,
    "get_zeros_int32": get_zeros_int32,
    "get_zeros_int64": get_zeros_int64,
    "get_ones_int32": get_ones_int32,
    "get_small_rand_int32": get_small_rand_int32,
    "get_zeros_then_ones": get_zeros_then_ones,
    "get_sentences": get_sentences,
}


OpsetConstraint = namedtuple("OpsetConstraint", "domain, min_version, max_version, excluded_version")


class Test(object):
    """Main Test class."""

    cache_dir = None
    target = []

    def __init__(self, url, local, input_func, input_names, output_names,
                 disabled=False, rtol=0.01, atol=1e-6, ptol=0, dequantize=False,
                 check_only_shape=False, model_type="frozen", force_input_shape=False,
                 skip_tensorflow=False, opset_constraints=None, tf_min_version=None, tag=None,
                 skip_conversion=False, converted_model=None, signature_def=None, concrete_function=None,
                 large_model=False, structured_outputs=None, run_tf_frozen=None, use_custom_ops=False,
                 ort_profile=None, tf_profile=None):
        self.url = url
        self.input_func = input_func
        self.local = local
        self.input_names = input_names
        self.output_names = output_names
        self.disabled = disabled
        self.large_model = large_model
        self.ort_profile = ort_profile
        self.tf_profile = tf_profile
        self.use_custom_ops = use_custom_ops
        if run_tf_frozen is None:
            run_tf_frozen = not self.large_model
        self.run_tf_frozen = run_tf_frozen
        self.structured_outputs = structured_outputs  # Needed to determine output order for tf_function
        self.rtol = rtol
        self.atol = atol
        self.ptol = ptol
        self.dequantize = dequantize
        self.check_only_shape = check_only_shape
        self.perf = None
        self.tf_runtime = 0
        self.onnx_runtime = 0
        self.model_type = model_type
        self.tag = tag
        self.force_input_shape = force_input_shape
        self.skip_tensorflow = skip_tensorflow
        self.skip_conversion = skip_conversion
        self.converted_model = converted_model
        self.opset_constraints = opset_constraints
        self.tf_min_version = tf_min_version
        self.signatures = [signature_def] if signature_def else None
        self.concrete_function = concrete_function

    def make_input(self, v):
        """Allows each input to specify its own function while defaulting to the input_get function"""
        if isinstance(v, dict):
            if "input_get" in v:
                return _INPUT_FUNC_MAPPING[v["input_get"]](v["shape"])
            if "value" in v:
                return np.array(v["value"])
        return self.input_func(v)

    def download_model(self):
        """Download model from url."""
        cache_dir = Test.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        url = self.url
        if url.startswith(r'module://'):
            return self.download_from_module()
        k = url.rfind('/')
        fname = self.url[k + 1:]
        dir_name = fname + "_dir"
        ftype = None
        if url.endswith(".tar.gz") or url.endswith(".tgz"):
            ftype = 'tgz'
            dir_name = fname.replace(".tar.gz", "").replace(".tgz", "")
        elif url.endswith('.zip'):
            ftype = 'zip'
            dir_name = fname.replace(".zip", "")
        elif url.endswith('.tflite'):
            ftype = 'tflite'
            dir_name = fname.replace(".tflite", "")
        elif self.model_type == 'tflite':
            fname = self.local
            dir_name = fname.replace(".tflite", "") + "_dir"
        elif self.model_type == 'tfjs':
            ftype = 'tgz'
            fname = 'model.tar.gz'
            dir_name = "_".join(url.split("/")[5:-3]) + "_dir"
        dir_name = os.path.join(cache_dir, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        fpath = os.path.join(dir_name, fname)
        if not os.path.exists(fpath):
            utils.get_url(url, fpath)
        model_path = os.path.join(dir_name, self.local)
        if not os.path.exists(model_path) or self.local == ".":
            if ftype == 'tgz':
                tar = tarfile.open(fpath)
                tar.extractall(dir_name)
                tar.close()
            elif ftype == 'zip':
                zip_ref = zipfile.ZipFile(fpath, 'r')
                zip_ref.extractall(dir_name)
                zip_ref.close()
        return fpath, dir_name

    def download_from_module(self):
        """Download a model from a python module"""
        cache_dir = Test.cache_dir
        from importlib import import_module
        i = self.url.rfind('//')
        module, model_name = self.url[i + 2:].split('/')
        mod_object = import_module(module)
        model_class = getattr(mod_object, model_name)
        model = model_class()
        fpath = os.path.join(cache_dir, self.local)
        model.save(fpath)
        return fpath, cache_dir

    def run_tensorflow(self, sess, inputs):
        """Run model on tensorflow so we have a reference output."""
        feed_dict = {}
        for k, v in inputs.items():
            k = sess.graph.get_tensor_by_name(k)
            feed_dict[k] = v
        logger.info("Running TF")
        result = sess.run(self.output_names, feed_dict=feed_dict)
        if self.perf:
            logger.info("Running TF perf")
            n = 0
            start = time.time()
            stop = start + PERF_TIME
            while time.time() < stop:
                for _ in range(PERF_STEP):
                    _ = sess.run(self.output_names, feed_dict=feed_dict)
                n += PERF_STEP
            self.tf_runtime = 1000 * (time.time() - start) / n
            logger.info("TF perf {:.2f}ms/inference, n={}".format(self.tf_runtime, n))
        return result

    def to_onnx(self, tf_graph, opset=None, extra_opset=None, shape_override=None, input_names=None,
                const_node_values=None, initialized_tables=None, tflite_path=None, tensors_to_rename=None,
                tfjs_path=None):
        """Convert graph to tensorflow."""
        if extra_opset is None:
            extra_opset = []
        if self.use_custom_ops:
            extra_opset.append(utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1))
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset,
                                extra_opset=extra_opset, target=Test.target, shape_override=shape_override,
                                input_names=input_names, output_names=self.output_names,
                                const_node_values=const_node_values, initialized_tables=initialized_tables,
                                tflite_path=tflite_path, dequantize=self.dequantize,
                                tensors_to_rename=tensors_to_rename, tfjs_path=tfjs_path)

    def run_onnxruntime(self, name, model_proto, inputs, outputs, external_tensor_storage=None):
        """Run test against onnxruntime backend."""
        import onnxruntime as rt
        model_path = utils.save_onnx_model(TEMP_DIR, name, inputs, model_proto, include_test_data=True,
                                           as_text=utils.is_debug_mode(),
                                           external_tensor_storage=external_tensor_storage)
        logger.info("Model saved to %s", model_path)
        providers = ['CPUExecutionProvider']
        if rt.get_device() == "GPU":
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus is None or len(gpus) > 1:
                providers = ['CUDAExecutionProvider']

        opt = rt.SessionOptions()
        if self.use_custom_ops:
            from onnxruntime_extensions import get_library_path
            opt.register_custom_ops_library(get_library_path())
        if self.ort_profile is not None:
            opt.enable_profiling = True
        m = rt.InferenceSession(model_path, sess_options=opt, providers=providers)
        results = m.run(outputs, inputs)
        if self.perf:
            n = 0
            start = time.time()
            stop = start + PERF_TIME
            while time.time() < stop:
                for _ in range(PERF_STEP):
                    _ = m.run(outputs, inputs)
                n += PERF_STEP
            self.onnx_runtime = 1000 * (time.time() - start) / n
            logger.info("ORT perf {:.2f}ms/inference, n={}".format(self.onnx_runtime, n))
        if self.ort_profile is not None:
            tmp_path = m.end_profiling()
            shutil.move(tmp_path, self.ort_profile)
        return results

    @staticmethod
    def create_onnx_file(name, model_proto, inputs, outdir, external_tensor_storage=None):
        os.makedirs(outdir, exist_ok=True)
        if external_tensor_storage is None:
            model_path = os.path.join(outdir, name + ".onnx")
            utils.save_protobuf(model_path, model_proto)
        else:
            model_path = os.path.join(outdir, name + ".zip")
            utils.save_onnx_zip(model_path, model_proto, external_tensor_storage)
        logger.info("Created %s", model_path)

    def run_test(self, name, backend="onnxruntime", onnx_file=None, opset=None, extra_opset=None, perf=None):
        """Run complete test against backend."""
        self.perf = perf

        # get the model
        if self.url:
            _, dir_name = self.download_model()
            logger.info("Downloaded to %s", dir_name)
            model_path = os.path.join(dir_name, self.local) if self.local != "." else dir_name
        else:
            model_path = self.local

        logger.info("Load model from %s", model_path)
        tf_reset_default_graph()
        tf.keras.backend.clear_session()
        input_names = list(self.input_names.keys())
        initialized_tables = {}
        outputs = self.output_names
        tflite_path = None
        tfjs_path = None
        to_rename = {}
        if self.model_type in ["checkpoint"]:
            graph_def, input_names, outputs = tf_loader.from_checkpoint(model_path, input_names, outputs)
        elif self.model_type in ["saved_model"]:
            loaded = tf_loader.from_saved_model(model_path, None, None, self.tag, self.signatures,
                                                self.concrete_function, self.large_model,
                                                return_concrete_func=not self.run_tf_frozen,
                                                return_initialized_tables=True, return_tensors_to_rename=True)
            if not self.run_tf_frozen:
                # Must maintain ref to imported since concrete_func uses weak refs
                # pylint: disable=unused-variable
                graph_def, input_names, outputs, concrete_func, imported, initialized_tables, to_rename = loaded
            else:
                graph_def, input_names, outputs, initialized_tables, to_rename = loaded
        elif self.model_type in ["keras"]:
            graph_def, input_names, outputs = tf_loader.from_keras(model_path, input_names, outputs)
        elif self.model_type in ["tflite"]:
            tflite_path = model_path
            graph_def = None
        elif self.model_type in ["tfjs"]:
            tfjs_path = model_path
            graph_def = None
        else:
            graph_def, input_names, outputs = tf_loader.from_graphdef(model_path, input_names, outputs)

        if utils.is_debug_mode():
            utils.save_protobuf(os.path.join(TEMP_DIR, name + "_after_tf_optimize.pb"), graph_def)

        if tflite_path is not None:
            inputs = {}
            for k in input_names:
                v = self.input_names[k]
                inputs[k] = self.make_input(v)

            interpreter = tf.lite.Interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_name_to_index = {n['name'].split(':')[0]: n['index'] for n in input_details}
            for k, v in inputs.items():
                interpreter.resize_tensor_input(input_name_to_index[k], v.shape)
            interpreter.allocate_tensors()
            def run_tflite():
                for k, v in inputs.items():
                    interpreter.set_tensor(input_name_to_index[k], v)
                interpreter.invoke()
                result = [interpreter.get_tensor(output['index']) for output in output_details]
                return result
            if not self.skip_tensorflow:
                tf_results = run_tflite()
                if self.perf:
                    logger.info("Running TFLite perf")
                    n = 0
                    start = time.time()
                    stop = start + PERF_TIME
                    while time.time() < stop:
                        for _ in range(PERF_STEP):
                            _ = run_tflite()
                        n += PERF_STEP
                    self.tf_runtime = 1000 * (time.time() - start) / n
                    logger.info("TFLite perf {:.2f}ms/inference, n={}".format(self.tf_runtime, n))
                logger.info("TFLite OK")

        if tfjs_path is not None:
            inputs = {}
            for k in input_names:
                v = self.input_names[k]
                inputs[k] = self.make_input(v)
            if not self.skip_tensorflow:
                logger.info("Running TFJS")
                tf_results = run_tfjs(tfjs_path, inputs, outputs)
                logger.info("TFJS OK")

        if not self.run_tf_frozen:
            inputs = {}
            for k in input_names:
                v = self.input_names[k]
                inputs[to_rename.get(k, k)] = tf.constant(self.make_input(v))
            tf_func = tf.function(concrete_func)
            logger.info("Running TF")
            tf_results_d = tf_func(**inputs)
            # If there is only a single output a dict might not be returned
            if isinstance(tf_results_d, tf.Tensor):
                tf_results = [tf_results_d]
            else:
                tf_results = [tf_results_d[k] for k in sorted(tf_results_d.keys())]
            tf_results = [tf_res.numpy() for tf_res in tf_results]
            if self.perf:
                logger.info("Running TF perf")
                n = 0
                start = time.time()
                stop = start + PERF_TIME
                if self.tf_profile is not None:
                    tf.profiler.experimental.start(self.tf_profile)
                while time.time() < stop:
                    for _ in range(PERF_STEP):
                        _ = concrete_func(**inputs)
                    n += PERF_STEP
                if self.tf_profile is not None:
                    tf.profiler.experimental.stop()
                self.tf_runtime = 1000 * (time.time() - start) / n
                logger.info("TF perf {:.2f}ms/inference, n={}".format(self.tf_runtime, n))
            logger.info("TensorFlow OK")

        const_node_values = None
        tf_graph = None

        if graph_def is not None:
            inputs = {}
            tf_reset_default_graph()

            with tf.Graph().as_default() as tf_graph:
                from tf2onnx.tf_utils import compress_graph_def
                if self.large_model:
                    const_node_values = compress_graph_def(graph_def)
                tf.import_graph_def(graph_def, name='')

            with tf_session(graph=tf_graph) as sess:
                # create the input data
                for k in input_names:
                    v = self.input_names[k]
                    t = sess.graph.get_tensor_by_name(k)
                    expected_dtype = tf.as_dtype(t.dtype).name
                    if isinstance(v, six.text_type) and v.startswith("np."):
                        np_value = eval(v)  # pylint: disable=eval-used
                        if expected_dtype != np_value.dtype:
                            logger.warning("dtype mismatch for input %s: expected=%s, actual=%s", k, expected_dtype,
                                           np_value.dtype)
                        inputs[k] = np_value.astype(expected_dtype)
                    else:
                        if expected_dtype == "string":
                            inputs[k] = self.make_input(v).astype(np.str).astype(object)
                        else:
                            inputs[k] = self.make_input(v).astype(expected_dtype)

                # run the model with tensorflow
                if self.skip_tensorflow:
                    logger.info("TensorFlow SKIPPED")
                elif self.run_tf_frozen:
                    if self.tf_profile is not None:
                        tf.profiler.experimental.start(self.tf_profile)
                    tf_results = self.run_tensorflow(sess, inputs)
                    if self.tf_profile is not None:
                        tf.profiler.experimental.stop()
                    logger.info("TensorFlow OK")
                tf_graph = sess.graph


        model_proto = None
        if self.skip_conversion:
            if self.large_model:
                external_tensor_storage = ExternalTensorStorage()
                model_proto = utils.model_proto_from_zip(self.converted_model, external_tensor_storage)
            else:
                external_tensor_storage = None
                model_proto = utils.model_proto_from_file(self.converted_model)
            logger.info("ONNX loaded from file")
        else:
            try:
                # convert model to onnx
                if self.force_input_shape:
                    shape_override = {k: list(v.shape) for k, v in inputs.items()}
                else:
                    shape_override = None
                onnx_graph = self.to_onnx(tf_graph, opset=opset, extra_opset=extra_opset,
                                          shape_override=shape_override, input_names=inputs.keys(),
                                          const_node_values=const_node_values,
                                          initialized_tables=initialized_tables, tflite_path=tflite_path,
                                          tensors_to_rename=to_rename, tfjs_path=tfjs_path)
                onnx_graph = optimizer.optimize_graph(onnx_graph)
                print("ONNX", onnx_graph.dump_node_statistics())
                external_tensor_storage = ExternalTensorStorage() if self.large_model else None
                model_proto = onnx_graph.make_model("converted from tf2onnx",
                                                    external_tensor_storage=external_tensor_storage)
                logger.info("To_ONNX, OK")
                if onnx_file:
                    self.create_onnx_file(name, model_proto, inputs, onnx_file, external_tensor_storage)
                if self.converted_model:
                    if self.large_model:
                        utils.save_onnx_zip(self.converted_model, model_proto, external_tensor_storage)
                    else:
                        utils.save_protobuf(self.converted_model, model_proto)
                    logger.info("Created %s", self.converted_model)

            except Exception:
                logger.error("To_ONNX FAIL", exc_info=1)
                return False

        try:
            onnx_results = None
            if backend == "onnxruntime":
                struc_outputs = [to_rename.get(k, k) for k in self.output_names]
                struc_inputs = {to_rename.get(k, k): v for k, v in inputs.items()}
                onnx_results = self.run_onnxruntime(
                    name, model_proto, struc_inputs, struc_outputs, external_tensor_storage)
            else:
                raise ValueError("unknown backend")
            logger.info("Run_ONNX OK")

            try:
                if self.skip_tensorflow:
                    logger.info("Results: skipped tensorflow")
                else:
                    if self.check_only_shape:
                        for tf_res, onnx_res in zip(tf_results, onnx_results):
                            np.testing.assert_array_equal(tf_res.shape, onnx_res.shape)
                    else:
                        for tf_res, onnx_res in zip(tf_results, onnx_results):
                            good_cnt = np.count_nonzero(np.isclose(tf_res, onnx_res, rtol=self.rtol, atol=self.atol))
                            bad_cnt = tf_res.size - good_cnt
                            if bad_cnt > self.ptol * tf_res.size:
                                # Prints a nice error message with stats
                                np.testing.assert_allclose(tf_res, onnx_res, rtol=self.rtol, atol=self.atol)
                    logger.info("Results: OK")
                return True
            except Exception:
                logger.error("Results", exc_info=1)

        except Exception:
            logger.error("Run_ONNX FAIL", exc_info=1)

        return False

    def check_opset_constraints(self, opset, extra_opset=None):
        """ Return (condition, reason) tuple, condition is True if constraints are met. """
        if not self.opset_constraints:
            return True, None

        opsets = {"onnx": opset}
        if extra_opset:
            for e in extra_opset:
                opsets[e.domain] = e.version

        for constraint in self.opset_constraints:
            domain = constraint.domain
            opset_version = opsets.get(domain)
            if not opset_version:
                return False, "conversion requires opset {}".format(domain)

            if constraint.min_version and opset_version < constraint.min_version:
                reason = "conversion requires opset {} >= {}".format(domain, constraint.min_version)
                return False, reason

            if constraint.max_version and opset_version > constraint.max_version:
                reason = "conversion requires opset {} <= {}".format(domain, constraint.max_version)
                return False, reason

            if constraint.excluded_version:
                if utils.is_list_or_tuple(constraint.excluded_version):
                    skip = opset_version in constraint.excluded_version
                else:
                    skip = opset_version == constraint.excluded_version
                if skip:
                    reason = "conversion requires opset {} != {}".format(domain, constraint.excluded_version)
                    return False, reason

        return True, None


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), 'pre-trained'),
                        help="pre-trained models cache dir")
    parser.add_argument("--config", default="tests/run_pretrained_models.yaml", help="yaml config to use")
    parser.add_argument("--tests", help="tests to run")
    parser.add_argument("--target", default="", help="target platform")
    parser.add_argument("--backend", default="onnxruntime",
                        choices=["onnxruntime"], help="backend to use")
    parser.add_argument("--opset", type=int, default=None, help="opset to use")
    parser.add_argument("--extra_opset", default=None,
                        help="extra opset with format like domain:version, e.g. com.microsoft:1")
    parser.add_argument("--skip_tf_tests", help="skip non-tflite tests", default="False")
    parser.add_argument("--skip_tflite_tests", help="skip tflite tests", default="False")
    parser.add_argument("--skip_tfjs_tests", help="skip tfjs tests", default="False")
    parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--list", help="list tests", action="store_true")
    parser.add_argument("--onnx-file", help="create onnx file in directory")
    parser.add_argument("--perf", help="capture performance numbers")
    parser.add_argument("--include-disabled", help="include disabled tests", action="store_true")
    args = parser.parse_args()

    args.target = args.target.split(",")
    args.skip_tf_tests = args.skip_tf_tests.upper() == "TRUE"
    args.skip_tflite_tests = args.skip_tflite_tests.upper() == "TRUE"
    args.skip_tfjs_tests = args.skip_tfjs_tests.upper() == "TRUE"
    if args.extra_opset:
        tokens = args.extra_opset.split(':')
        if len(tokens) != 2:
            raise ValueError("invalid extra_opset argument")
        args.extra_opset = [utils.make_opsetid(tokens[0], int(tokens[1]))]
    return args


def load_tests_from_yaml(path):
    """Create test class from yaml file."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)

    tests = {}
    config = yaml.safe_load(open(path, 'r').read())
    for name, settings in config.items():
        if name in tests:
            raise ValueError("Found duplicated test: {}".format(name))

        # parse model and url, non-absolute local path is relative to yaml directory
        model = settings.get("model")
        url = settings.get("url")
        if not url and not os.path.isabs(model):
            model = os.path.join(base_dir, model)

        # parse input_get
        input_func = settings.get("input_get")
        input_func = _INPUT_FUNC_MAPPING[input_func]

        # parse inputs, non-absolute npy file path for np.load is relative to yaml directory
        inputs = settings.get("inputs")
        for k, v in list(inputs.items()):
            if isinstance(v, str):
                # assume at most 1 match
                matches = re.findall(r"np\.load\((r?['\"].*?['\"])", v)
                if matches:
                    npy_path = matches[0].lstrip('r').strip("'").strip('"')
                    if not os.path.isabs(npy_path):
                        abs_npy_path = os.path.join(base_dir, npy_path)
                        inputs[k] = v.replace(matches[0], "r'{}'".format(abs_npy_path))

        # parse opset_constraints
        opset_constraints = []
        section = settings.get("opset_constraints")
        if section:
            for k, v in section.items():
                c = OpsetConstraint(k, min_version=v.get("min"), max_version=v.get("max"),
                                    excluded_version=v.get("excluded"))
                opset_constraints.append(c)

        kwargs = {}
        for kw in ["rtol", "atol", "ptol", "disabled", "check_only_shape", "model_type", "concrete_function",
                   "skip_tensorflow", "force_input_shape", "tf_min_version", "tag", "skip_conversion",
                   "converted_model", "signature_def", "large_model", "structured_outputs", "run_tf_frozen",
                   "use_custom_ops", "dequantize", "ort_profile", "tf_profile"]:
            if settings.get(kw) is not None:
                kwargs[kw] = settings[kw]

        test = Test(url, model, input_func, inputs, settings.get("outputs"),
                    opset_constraints=opset_constraints, **kwargs)
        tests[name] = test
    return tests


def main():
    args = get_args()
    logging.basicConfig(level=logging.get_verbosity_level(args.verbose))
    if args.debug:
        utils.set_debug_mode(True)

    Test.cache_dir = args.cache
    Test.target = args.target
    tests = load_tests_from_yaml(args.config)
    if args.list:
        logger.info(sorted(tests.keys()))
        return 0
    if args.tests:
        test_keys = args.tests.split(",")
    else:
        test_keys = list(tests.keys())

    failed = 0
    count = 0
    for test in test_keys:
        logger.info("===================================")

        t = tests[test]
        if args.tests is None:
            if t.disabled and not args.include_disabled:
                logger.info("Skip %s: disabled", test)
                continue

            if args.skip_tfjs_tests and t.model_type == "tfjs":
                logger.info("Skip %s: tfjs test", test)
                continue
            if args.skip_tflite_tests and t.model_type == "tflite":
                logger.info("Skip %s: tflite test", test)
                continue
            if args.skip_tf_tests and t.model_type not in ["tflite", "tfjs"]:
                logger.info("Skip %s: tf test", test)
                continue

            condition, reason = t.check_opset_constraints(args.opset, args.extra_opset)
            if not condition:
                logger.info("Skip %s: %s", test, reason)
                continue

            if t.tf_min_version:
                if tf_utils.get_tf_version() < Version(str(t.tf_min_version)):
                    logger.info("Skip %s: %s %s", test, "Min TF version needed:", t.tf_min_version)
                    continue

        count += 1
        try:
            logger.info("Running %s", test)
            ret = t.run_test(test, backend=args.backend, onnx_file=args.onnx_file,
                             opset=args.opset, extra_opset=args.extra_opset, perf=args.perf)
        except Exception:
            logger.error("Failed to run %s", test, exc_info=1)
            ret = None
        finally:
            if not utils.is_debug_mode():
                utils.delete_directory(TEMP_DIR)
        if not ret:
            failed += 1

    logger.info("===================================")
    logger.info("RESULT: %s failed of %s, backend=%s", failed, count, args.backend)

    if args.perf:
        with open(args.perf, "w") as f:
            f.write("test,tensorflow,onnx\n")
            for test in test_keys:
                t = tests[test]
                if t.perf:
                    # Report perf in ms per inference
                    f.write("{},{},{}\n".format(test, t.tf_runtime, t.onnx_runtime))
    return failed


if __name__ == "__main__":
    sys.exit(main())
