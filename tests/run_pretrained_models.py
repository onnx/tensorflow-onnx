# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import tarfile
import tempfile
import urllib
import urllib.request
import zipfile

import PIL.Image
import numpy as np
import tensorflow as tf
import tf2onnx
import yaml
from tensorflow.core.framework import graph_pb2
from tf2onnx.tfonnx import process_tf_graph

TMPPATH = tempfile.mkdtemp()


def get_beach(inputs):
    """Get beach image as input."""
    for name, shape in inputs.items():
        break
    resize_to = shape[1:3]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beach.jpg")
    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.reshape(shape)
    return {name: img_np}


def get_random(inputs):
    """Get random input."""
    d = {}
    for k, v in inputs.items():
        d[k] = np.random.sample(v).astype(np.float32)
    return d


def get_random256(inputs):
    """Get random imput between 0 and 255."""
    d = {}
    for k, v in inputs.items():
        d[k] = np.round(np.random.sample(v) * 256).astype(np.float32)
    return d


def get_ramp(inputs):
    """Get ramp input."""
    d = {}
    for k, v in inputs.items():
        size = np.prod(v)
        d[k] = np.linspace(1, size, size).reshape(v).astype(np.float32)
    return d


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


class Test(object):
    cache_dir = None

    def __init__(self, url, local, make_input, input_names, output_names,
                 disabled=False, more_inputs=None, rtol=0.01, atol=0., check_only_shape=False):
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
            urllib.request.urlretrieve(url, fpath)
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
        return result

    @staticmethod
    def to_onnx(tf_graph, opset=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, opset=opset)

    def run_caffe2(self, name, onnx_graph, inputs):
        """Run test again caffe2 backend."""
        import caffe2.python.onnx.backend
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        prepared_backend = caffe2.python.onnx.backend.prepare(model_proto)
        results = prepared_backend.run(inputs)
        return results

    def run_onnxmsrt(self, name, onnx_graph, inputs):
        """Run test against onnxmsrt backend."""
        import lotus
        # create model and datafile in tmp path.
        model_path = os.path.join(TMPPATH, name + "_model.pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = lotus.ModelExecutor(model_path)
        results = m.run(self.output_names, inputs)
        return results

    def run_onnxmsrtnext(self, name, onnx_graph, inputs):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, name + ".pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = lotus.InferenceSession(model_path)
        results = m.run(self.output_names, inputs)
        return results

    def run_onnxcntk(self, name, onnx_graph, inputs):
        """Run test against cntk backend."""
        import cntk as C
        model_path = os.path.join(TMPPATH, name + "_model.pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        z = C.Function.load(model_path, format=C.ModelFormat.ONNX)
        input_args = {}
        # FIXME: the model loads but eval() throws
        for arg in z.arguments:
            input_args[arg] = inputs[arg.name]
        results = z.eval(input_args)
        return results

    def create_onnx_file(self, name, onnx_graph, inputs, outdir):
        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, name + ".onnx")
        model_proto = onnx_graph.make_model(name, inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("\tcreated", model_path)

    def run_test(self, name, backend="caffe2", debug=False, onnx_file=None, opset=None):
        """Run complete test against backend."""
        print(name)
        if self.url:
            _, dir_name = self.download_file()
            model_path = os.path.join(dir_name, self.local)
        else:
            model_path = self.local
        print("\tdownloaded", model_path)

        inputs = self.make_input(self.input_names)
        if self.more_inputs:
            for k, v in self.more_inputs.items():
                inputs[k] = v
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        g = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=g) as sess:
            tf_results = self.run_tensorflow(sess, inputs)
            onnx_graph = None
            print("\ttensorflow", "OK")
            try:
                onnx_graph = self.to_onnx(sess.graph, opset=opset)
                print("\tto_onnx", "OK")
                if debug:
                    onnx_graph.dump_graph()
                if onnx_file:
                    self.create_onnx_file(name, onnx_graph, inputs, onnx_file)
            except Exception as ex:
                print("\tto_onnx", "FAIL", ex)

        try:
            onnx_results = None
            if backend == "caffe2":
                onnx_results = self.run_caffe2(name, onnx_graph, inputs)
            elif backend == "onnxmsrt":
                onnx_results = self.run_onnxmsrt(name, onnx_graph, inputs)
            elif backend == "onnxmsrtnext":
                onnx_results = self.run_onnxmsrtnext(name, onnx_graph, inputs)
            elif backend == "cntk":
                onnx_results = self.run_onnxcntk(name, onnx_graph, inputs)
            else:
                raise ValueError("unknown backend")
            print("\trun_onnx OK")

            try:
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
    parser.add_argument("--backend", default="caffe2",
                        choices=["caffe2", "onnxmsrt", "onnxmsrtnext", "cntk"], help="backend to use")
    parser.add_argument("--verbose", help="verbose output", action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="opset to use")
    parser.add_argument("--debug", help="debug vlog", action="store_true")
    parser.add_argument("--list", help="list tests", action="store_true")
    parser.add_argument("--onnx-file", help="create onnx file in directory")
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
        for kw in ["rtol", "atol", "disabled", "more_inputs", "check_only_shape"]:
            if v.get(kw) is not None:
                kwargs[kw] = v[kw]

        test = Test(v.get("url"), v.get("model"), input_func, v.get("inputs"), v.get("outputs"), **kwargs)
        tests[k] = test
    return tests


def main():
    args = get_args()
    Test.cache_dir = args.cache
    tf2onnx.utils.ONNX_UNKNOWN_DIMENSION = 1
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
            ret = t.run_test(test, backend=args.backend, debug=args.debug, onnx_file=args.onnx_file, opset=args.opset)
        except Exception as ex:
            ret = None
            print(ex)
        if not ret:
            failed += 1

    print("=== RESULT: {} failed of {}, backend={}".format(failed, count, args.backend))


if __name__ == "__main__":
    main()
