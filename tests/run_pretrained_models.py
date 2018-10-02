# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tarfile
import time
import tempfile
import requests
import zipfile

import PIL.Image
import numpy as np
import tensorflow as tf
import tf2onnx
import yaml
from tensorflow.core.framework import graph_pb2
from tf2onnx.tfonnx import process_tf_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils

from tf2onnx.optimizer.transpose_optimizer import TransposeOptimizer

TMPPATH = tempfile.mkdtemp()
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


def node_name(name):
    """Get node name without io#."""
    assert isinstance(name, str)
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


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


class Test(object):
    cache_dir = None

    def __init__(self, url, local, make_input, input_names, output_names,
                 disabled=False, more_inputs=None, rtol=0.01, atol=0.,
                 check_only_shape=False, model_type="frozen", force_input_shape=False,
                 skip_tensorflow=False):
        self.url = url
        self.make_input = make_input
        self.local = local
        self.input_names = input_names
        self.output_names = output_names
        self.disabled = disabled
        self.more_inputs = more_inputs
        self.rtol = rtol
        self.atol = atol
        self.check_only_shape = check_only_shape
        self.perf = None
        self.tf_runtime = 0
        self.onnx_runtime = 0
        self.model_type = model_type
        self.force_input_shape = force_input_shape
        self.skip_tensorflow = skip_tensorflow

    def download_file(self):
        """Download file from url."""
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
            response = requests.get(url)
            if response.status_code not in [200]:
                response.raise_for_status()
            with open(fpath, "wb") as f:
                f.write(response.content)
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
        """Run model on tensorflow so we have a referecne output."""
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

    @staticmethod
    def to_onnx(tf_graph, opset=None, shape_override=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset, shape_override=shape_override)

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

    def run_onnxmsrtnext(self, name, model_proto, inputs):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, name + ".pb")
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = lotus.InferenceSession(model_path)
        results = m.run(self.output_names, inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = m.run(self.output_names, inputs)
            self.onnx_runtime = time.time() - start
        return results

    def run_onnxruntime(self, name, model_proto, inputs):
        """Run test against msrt-next backend."""
        import onnxruntime as rt
        model_path = os.path.join(TMPPATH, name + ".pb")
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = rt.InferenceSession(model_path)
        results = m.run(self.output_names, inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = m.run(self.output_names, inputs)
            self.onnx_runtime = time.time() - start
        return results

    def create_onnx_file(self, name, model_proto, inputs, outdir):
        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, name + ".onnx")
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("\tcreated", model_path)

    def run_test(self, name, backend="caffe2", debug=False, onnx_file=None, opset=None, perf=None, transpose_opt=None):
        """Run complete test against backend."""
        print(name)
        self.perf = perf

        # get the model
        if self.url:
            _, dir_name = self.download_file()
            model_path = os.path.join(dir_name, self.local)
        else:
            model_path = self.local
            dir_name = os.path.dirname(self.local)
        print("\tdownloaded", model_path)

        if self.model_type in ["checkpoint"]:
            #
            # if the input model is a checkpoint, convert it to a frozen model
            saver = tf.train.import_meta_graph(model_path)
            with tf.Session() as sess:
                saver.restore(sess, model_path[:-5])
                frozen_graph = freeze_session(sess, output_names=self.output_names)
                tf.train.write_graph(frozen_graph, dir_name, "frozen.pb", as_text=False)
            model_path = os.path.join(dir_name, "frozen.pb")
        elif self.model_type in ["saved_model"]:
            # saved_model format - convert to checkpoint
            with tf.Session() as sess:
                meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
                inputs = {}
                outputs = {}
                for k in meta_graph_def.signature_def.keys():
                    inputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, k).inputs
                    for input_key, input_tensor in sorted(inputs_tensor_info.items()):
                        inputs[input_tensor.name] = sess.graph.get_tensor_by_name(input_tensor.name)
                    outputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, k).outputs
                    for output_key, output_tensor in sorted(outputs_tensor_info.items()):
                        outputs[output_tensor.name] = sess.graph.get_tensor_by_name(output_tensor.name)
                frozen_graph = freeze_session(sess, output_names=list(outputs.keys()))
                tf.train.write_graph(frozen_graph, dir_name, "frozen.pb", as_text=False)
            model_path = os.path.join(dir_name, "frozen.pb")

        # create the input data
        inputs = {}
        for k, v in self.input_names.items():
            if isinstance(v, str) and v.startswith("np."):
                inputs[k] = eval(v)
            else:
                inputs[k] = self.make_input(v)
        if self.more_inputs:
            for k, v in self.more_inputs.items():
                inputs[k] = v
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph_def = tf2onnx.tfonnx.tf_optimize(None, inputs, self.output_names, graph_def, transpose_opt)
        shape_override = {}
        g = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=g) as sess:

            # fix inputs if needed
            for k in inputs.keys():
                t = sess.graph.get_tensor_by_name(k)
                dtype = tf.as_dtype(t.dtype).name
                if type != "float32":
                    v = inputs[k]
                    inputs[k] = v.astype(dtype)
            if self.force_input_shape:
                shape_override = self.input_names

            # run the model with tensorflow
            if self.skip_tensorflow:
                print("\ttensorflow", "SKIPPED")
            else:
                tf_results = self.run_tensorflow(sess, inputs)
                print("\ttensorflow", "OK")
            model_proto = None
            try:
                # convert model to onnx
                onnx_graph = self.to_onnx(sess.graph, opset=opset, shape_override=shape_override)
                if transpose_opt:
                    optimizer = TransposeOptimizer(onnx_graph, debug)
                    optimizer.optimize()
  
                model_proto = onnx_graph.make_model("test", self.output_names)
                print("\tto_onnx", "OK")
                if debug:
                    model_proto.dump_graph()
                if onnx_file:
                    self.create_onnx_file(name, model_proto, inputs, onnx_file)
            except Exception as ex:
                print("\tto_onnx", "FAIL", ex)

        try:
            onnx_results = None
            if backend == "caffe2":
                onnx_results = self.run_caffe2(name, model_proto, inputs)
            elif backend == "onnxmsrtnext":
                onnx_results = self.run_onnxmsrtnext(name, model_proto, inputs)
            elif backend == "onnxruntime":
                onnx_results = self.run_onnxruntime(name, model_proto, inputs)
            else:
                raise ValueError("unknown backend")
            print("\trun_onnx OK")

            try:
                if self.skip_tensorflow:
                    print("\tResults: skipped tensorflow")
                else:
                    if self.check_only_shape:
                        for i in range(len(tf_results)):
                            np.testing.assert_array_equal(tf_results[i].shape, onnx_results[i].shape)
                    else:
                        for i in range(len(tf_results)):
                            np.testing.assert_allclose(tf_results[i], onnx_results[i], rtol=self.rtol, atol=self.atol)
                    print("\tResults: OK")
                return True
            except Exception as ex:
                print("\tResults: ", ex)

        except Exception as ex:
            print("\trun_onnx", "FAIL", ex)

        return False


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/tmp/pre-trained", help="pre-trained models cache dir")
    parser.add_argument("--config", default="tests/run_pretrained_models.yaml", help="yaml config to use")
    parser.add_argument("--tests", help="tests to run")
    parser.add_argument("--backend", default="onnxruntime",
                        choices=["caffe2", "onnxmsrtnext", "onnxruntime"], help="backend to use")
    parser.add_argument("--verbose", help="verbose output", action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="opset to use")
    parser.add_argument("--debug", help="debug vlog", action="store_true")
    parser.add_argument("--list", help="list tests", action="store_true")
    parser.add_argument("--onnx-file", help="create onnx file in directory")
    parser.add_argument("--perf", help="capture performance numbers")
    parser.add_argument("--optimize_transpose", help="eliminate transposes that can be removed", action="store_true")
    parser.add_argument("--include-disabled", help="include disabled tests", action="store_true")
    args = parser.parse_args()
    return args


def tests_from_yaml(fname):
    tests = {}
    config = yaml.load(open(fname, 'r').read())
    for k, v in config.items():
        input_func = v.get("input_get")
        input_func = _INPUT_FUNC_MAPPING[input_func]
        kwargs = {}
        for kw in ["rtol", "atol", "disabled", "more_inputs", "check_only_shape", "model_type",
                   "skip_tensorflow", "force_input_shape"]:
            if v.get(kw) is not None:
                kwargs[kw] = v[kw]

        test = Test(v.get("url"), v.get("model"), input_func, v.get("inputs"), v.get("outputs"), **kwargs)
        tests[k] = test
    return tests


def main():
    args = get_args()
    Test.cache_dir = args.cache
    tests = tests_from_yaml(args.config)
    if args.list:
        print(sorted(tests.keys()))
        return
    if args.tests:
        test_keys = args.tests.split(",")
    else:
        test_keys = list(tests.keys())

    failed = 0
    count = 0
    for test in test_keys:
        t = tests[test]
        if args.tests is None and t.disabled and not args.include_disabled:
            continue
        count += 1
        try:
            ret = t.run_test(test, backend=args.backend, debug=args.debug, onnx_file=args.onnx_file,
                             opset=args.opset, perf=args.perf, transpose_opt=args.optimize_transpose)
        except Exception as ex:
            ret = None
            print(ex)
        if not ret:
            failed += 1

    print("=== RESULT: {} failed of {}, backend={}".format(failed, count, args.backend))

    if args.perf:
        with open(args.perf, "w") as f:
            f.write("test,tensorflow,onnx\n")
            for test in test_keys:
                t = tests[test]
                if t.perf:
                    f.write("{},{},{}\n".format(test, t.tf_runtime, t.onnx_runtime))

if __name__ == "__main__":
    main()
