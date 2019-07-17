# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Tool to convert and test pre-trained tensorflow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import re
import sys
import tarfile
import time
import zipfile
from collections import namedtuple

import PIL.Image
import numpy as np
import six
import tensorflow as tf
# contrib ops are registered only when the module is imported, the following import statement is needed,
# otherwise tf runtime error will show up when the tf model is restored from pb file because of un-registered ops.
import tensorflow.contrib.rnn  # pylint: disable=unused-import
import yaml

import tf2onnx
from tf2onnx import loader, logging, optimizer, utils
from tf2onnx.tfonnx import process_tf_graph

# pylint: disable=broad-except,logging-not-lazy,unused-argument,unnecessary-lambda

logger = logging.getLogger("run_pretrained")

TEMP_DIR = os.path.join(utils.get_temp_directory(), "run_pretrained")
PERFITER = 1000


def get_beach(shape):
    """Get beach image as input."""
    resize_to = shape[1:3]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beach.jpg")
    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = np.stack([img_np] * shape[0], axis=0).reshape(shape)
    return img_np


def get_random(shape):
    """Get random input."""
    return np.random.sample(shape).astype(np.float32)


def get_random256(shape):
    """Get random imput between 0 and 255."""
    return np.round(np.random.sample(shape) * 256).astype(np.float32)


def get_ramp(shape):
    """Get ramp input."""
    size = np.prod(shape)
    return np.linspace(1, size, size).reshape(shape).astype(np.float32)


_INPUT_FUNC_MAPPING = {
    "get_beach": get_beach,
    "get_random": get_random,
    "get_random256": get_random256,
    "get_ramp": get_ramp
}

OpsetConstraint = namedtuple("OpsetConstraint", "domain, min_version, max_version, excluded_version")


class Test(object):
    """Main Test class."""

    cache_dir = None
    target = []

    def __init__(self, url, local, make_input, input_names, output_names,
                 disabled=False, rtol=0.01, atol=1e-6,
                 check_only_shape=False, model_type="frozen", force_input_shape=False,
                 skip_tensorflow=False, opset_constraints=None):
        self.url = url
        self.make_input = make_input
        self.local = local
        self.input_names = input_names
        self.output_names = output_names
        self.disabled = disabled
        self.rtol = rtol
        self.atol = atol
        self.check_only_shape = check_only_shape
        self.perf = None
        self.tf_runtime = 0
        self.onnx_runtime = 0
        self.model_type = model_type
        self.force_input_shape = force_input_shape
        self.skip_tensorflow = skip_tensorflow
        self.opset_constraints = opset_constraints

    def download_model(self):
        """Download model from url."""
        cache_dir = Test.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        url = self.url
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
        dir_name = os.path.join(cache_dir, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        fpath = os.path.join(dir_name, fname)
        if not os.path.exists(fpath):
            utils.get_url(url, fpath)
        model_path = os.path.join(dir_name, self.local)
        if not os.path.exists(model_path):
            if ftype == 'tgz':
                tar = tarfile.open(fpath)
                tar.extractall(dir_name)
                tar.close()
            elif ftype == 'zip':
                zip_ref = zipfile.ZipFile(fpath, 'r')
                zip_ref.extractall(dir_name)
                zip_ref.close()
        return fpath, dir_name

    def run_tensorflow(self, sess, inputs):
        """Run model on tensorflow so we have a reference output."""
        feed_dict = {}
        for k, v in inputs.items():
            k = sess.graph.get_tensor_by_name(k)
            feed_dict[k] = v
        result = sess.run(self.output_names, feed_dict=feed_dict)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = sess.run(self.output_names, feed_dict=feed_dict)
            self.tf_runtime = time.time() - start
        return result

    def to_onnx(self, tf_graph, opset=None, extra_opset=None, shape_override=None, input_names=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset,
                                extra_opset=extra_opset, target=Test.target, shape_override=shape_override,
                                input_names=input_names, output_names=self.output_names)

    def run_caffe2(self, name, model_proto, inputs):
        """Run test again caffe2 backend."""
        import caffe2.python.onnx.backend
        prepared_backend = caffe2.python.onnx.backend.prepare(model_proto)
        results = prepared_backend.run(inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = prepared_backend.run(inputs)
            self.onnx_runtime = time.time() - start
        return results

    def run_onnxruntime(self, name, model_proto, inputs):
        """Run test against onnxruntime backend."""
        import onnxruntime as rt
        model_path = utils.save_onnx_model(TEMP_DIR, name, inputs, model_proto, include_test_data=True,
                                           as_text=utils.is_debug_mode())
        logger.info("Model saved to %s", model_path)
        m = rt.InferenceSession(model_path)
        results = m.run(self.output_names, inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = m.run(self.output_names, inputs)
            self.onnx_runtime = time.time() - start
        return results

    @staticmethod
    def create_onnx_file(name, model_proto, inputs, outdir):
        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, name + ".onnx")
        utils.save_protobuf(model_path, model_proto)
        logger.info("Created %s", model_path)

    def run_test(self, name, backend="caffe2", onnx_file=None, opset=None, extra_opset=None,
                 perf=None, fold_const=None):
        """Run complete test against backend."""
        self.perf = perf

        # get the model
        if self.url:
            _, dir_name = self.download_model()
            logger.info("Downloaded to %s", dir_name)
            model_path = os.path.join(dir_name, self.local)
        else:
            model_path = self.local

        logger.info("Load model from %s", model_path)
        input_names = list(self.input_names.keys())
        outputs = self.output_names
        if self.model_type in ["checkpoint"]:
            graph_def, input_names, outputs = loader.from_checkpoint(model_path, input_names, outputs)
        elif self.model_type in ["saved_model"]:
            graph_def, input_names, outputs = loader.from_saved_model(model_path, input_names, outputs)
        else:
            graph_def, input_names, outputs = loader.from_graphdef(model_path, input_names, outputs)

        # remove unused input names
        input_names = list(set(input_names).intersection(self.input_names.keys()))
        graph_def = tf2onnx.tfonnx.tf_optimize(input_names, self.output_names, graph_def, fold_const)
        if utils.is_debug_mode():
            utils.save_protobuf(os.path.join(TEMP_DIR, name + "_after_tf_optimize.pb"), graph_def)

        inputs = {}
        shape_override = {}
        g = tf.import_graph_def(graph_def, name='')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g) as sess:
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
                    inputs[k] = self.make_input(v).astype(expected_dtype)

            if self.force_input_shape:
                for k, v in inputs.items():
                    shape_override[k] = list(v.shape)

            # run the model with tensorflow
            if self.skip_tensorflow:
                logger.info("TensorFlow SKIPPED")
            else:
                tf_results = self.run_tensorflow(sess, inputs)
                logger.info("TensorFlow OK")

            model_proto = None
            try:
                # convert model to onnx
                onnx_graph = self.to_onnx(sess.graph, opset=opset, extra_opset=extra_opset,
                                          shape_override=shape_override, input_names=inputs.keys())
                onnx_graph = optimizer.optimize_graph(onnx_graph)
                model_proto = onnx_graph.make_model("converted from tf2onnx")
                logger.info("To_ONNX, OK")
                if onnx_file:
                    self.create_onnx_file(name, model_proto, inputs, onnx_file)
            except Exception:
                logger.error("To_ONNX FAIL", exc_info=1)
                return False

        try:
            onnx_results = None
            if backend == "caffe2":
                onnx_results = self.run_caffe2(name, model_proto, inputs)
            elif backend == "onnxruntime":
                onnx_results = self.run_onnxruntime(name, model_proto, inputs)
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
    parser.add_argument("--cache", default="/tmp/pre-trained", help="pre-trained models cache dir")
    parser.add_argument("--config", default="tests/run_pretrained_models.yaml", help="yaml config to use")
    parser.add_argument("--tests", help="tests to run")
    parser.add_argument("--target", default="", help="target platform")
    parser.add_argument("--backend", default="onnxruntime",
                        choices=["caffe2", "onnxruntime"], help="backend to use")
    parser.add_argument("--opset", type=int, default=None, help="opset to use")
    parser.add_argument("--extra_opset", default=None,
                        help="extra opset with format like domain:version, e.g. com.microsoft:1")
    parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--list", help="list tests", action="store_true")
    parser.add_argument("--onnx-file", help="create onnx file in directory")
    parser.add_argument("--perf", help="capture performance numbers")
    parser.add_argument("--fold_const", help="enable tf constant_folding transformation before conversion",
                        action="store_true")
    parser.add_argument("--include-disabled", help="include disabled tests", action="store_true")
    args = parser.parse_args()

    args.target = args.target.split(",")
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
        for kw in ["rtol", "atol", "disabled", "check_only_shape", "model_type",
                   "skip_tensorflow", "force_input_shape"]:
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

            condition, reason = t.check_opset_constraints(args.opset, args.extra_opset)
            if not condition:
                logger.info("Skip %s: %s", test, reason)
                continue

        count += 1
        try:
            logger.info("Running %s", test)
            ret = t.run_test(test, backend=args.backend, onnx_file=args.onnx_file,
                             opset=args.opset, extra_opset=args.extra_opset, perf=args.perf,
                             fold_const=args.fold_const)
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
                    f.write("{},{},{}\n".format(test, t.tf_runtime, t.onnx_runtime))
    return failed


if __name__ == "__main__":
    sys.exit(main())
