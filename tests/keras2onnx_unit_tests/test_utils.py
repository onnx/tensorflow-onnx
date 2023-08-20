# SPDX-License-Identifier: Apache-2.0

import os
import sys
import onnx
from onnx import helper
import numpy as np
import mock_keras2onnx
from mock_keras2onnx.proto import keras, is_keras_older_than
from mock_keras2onnx.proto.tfcompat import is_tf2
from packaging.version import Version
from tf2onnx.keras2onnx_api import convert_keras, get_maximum_opset_supported
import time
import json
import urllib


# Mapping opset to ONNXRuntime version.
# TODO: upgrade opset to 18 for 1.14.0 after the below issue is fixed:
# https://github.com/onnx/tensorflow-onnx/issues/2132
ORT_OPSET_VERSION = {
    "1.6.0": 13, "1.7.0": 13, "1.8.0": 14, "1.9.0": 15, "1.10.0": 15, "1.11.0": 16,
    "1.12.0": 17, "1.13.0": 17, "1.14.0": 18, "1.15.0": 18
}

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')
test_level_0 = True


def convert_tf_crop_and_resize(scope, operator, container):
    if operator.target_opset < 11:
        raise ValueError("CropAndResize op is not supported for opset < 11")
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    mode_value = node.get_attr('method')
    transpose_node = oopb.apply_transpose(operator.inputs[0].full_name,
                                          name=operator.full_name + '_transpose_1',
                                          perm=[0, 3, 1, 2])
    cropandresize = oopb.add_node('CropAndResize',
                                  transpose_node + operator.input_full_names[1:],
                                  operator.full_name + '_crop_and_resize',
                                  op_domain='com.microsoft',
                                  op_version=1,
                                  mode=mode_value)
    oopb.apply_op_with_output("apply_transpose",
                              cropandresize,
                              operator.output_full_names,
                              name=operator.full_name + '_transpose_final',
                              perm=[0, 2, 3, 1])


# convert keras_contrib.layers.InstanceNormalization
def convert_InstanceNormalizationLayer(scope, operator, container):
    from mock_keras2onnx.common.onnx_ops import OnnxOperatorBuilder
    op = operator.raw_operator
    params = op.get_weights()
    assert len(op.input_shape) == 4
    beta = params[0].reshape(1, 1, 1, 1).astype('float32')
    gamma = params[1].reshape(1, 1, 1, 1).astype('float32')
    oopb = OnnxOperatorBuilder(container, scope)

    reducemean_1 = oopb.add_node('ReduceMean',
                                 [operator.inputs[0].full_name],
                                 operator.inputs[0].full_name + '_reduce_mean_1',
                                 axes=[1, 2, 3], keepdims=1)

    sub_1 = oopb.add_node('Sub',
                          [operator.inputs[0].full_name, reducemean_1],
                          operator.inputs[0].full_name + '_sub_1')

    mul = oopb.add_node('Mul',
                        [sub_1, sub_1],
                        operator.inputs[0].full_name + '_mul')

    reducemean_2 = oopb.add_node('ReduceMean',
                                 [mul],
                                 operator.inputs[0].full_name + '_reduce_mean_2',
                                 axes=[1, 2, 3], keepdims=1)

    sqrt = oopb.add_node('Sqrt',
                         [reducemean_2],
                         operator.inputs[0].full_name + '_sqrt')

    add = oopb.add_node('Add',
                        [sqrt,
                         ('_start', oopb.float, np.array([op.epsilon], dtype='float32'))],
                        operator.inputs[0].full_name + '_add')

    div = oopb.add_node('Div',
                        [sub_1, add],
                        operator.inputs[0].full_name + '_div')

    mul_scale = oopb.add_node('Mul',
                              [div,
                               ('_start', oopb.float, beta)],
                              operator.inputs[0].full_name + '_mul_scale')

    add_bias = oopb.add_node('Add',
                             [mul_scale,
                              ('_start', oopb.float, gamma)],
                             operator.inputs[0].full_name + '_add_bias')

    apply_identity(scope, add_bias, operator.outputs[0].full_name, container)


def print_mismatches(case_name, list_idx, expected_list, actual_list, rtol=1.e-3, atol=1.e-6):
    diff_list = abs(expected_list - actual_list)
    count_total = len(expected_list)
    count_error = 0
    count_current = 0

    for e_, a_, d_ in zip(expected_list, actual_list, diff_list):
        if d_ > atol + rtol * abs(a_):
            if count_error < 10:  # print the first 10 mismatches
                print(
                    "case = " + case_name + ", result mismatch for expected = " + str(e_) +
                    ", actual = " + str(a_) + " at location " + str(count_current), file=sys.stderr)
            count_error = count_error + 1
        count_current += 1

    print("case = " + case_name + ", " +
          str(count_error) + " mismatches out of " + str(count_total) + " for list " + str(list_idx),
          file=sys.stderr)


def load_profile_json(profile_file):
    print(f"loading profile output {profile_file} ...")

    with open(profile_file, "r") as f:
        sess_time = json.load(f)

    assert isinstance(sess_time, list)
    return sess_time


def parse_profile_results(sess_time, kernel_time_only=False, threshold=0):
    node_time = {}
    node_provider = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            if "provider" in item["args"]:
                device = "CPU" if item["args"]["provider"] == "CPUExecutionProvider" else "CUDA"
                if item["name"] not in node_provider:
                    node_provider[item["name"]] = device
                else:
                    assert node_provider[item["name"]] == device
            elif kernel_time_only:
                continue

            if item["name"] in node_time:
                node_time[item["name"]] += item["dur"]
            else:
                node_time[item["name"]] = item["dur"]
            total += item["dur"]

    results = []
    if (threshold > 0):
        results.append(f"Threshold of Percentage > {threshold:.2f}%")

    results.append(f"Duration\tPercentage\tProvider\tName")
    for k, v in sorted(node_time.items(), key=lambda x: x[1], reverse=True):
        provider = node_provider[k] if k in node_provider else ""
        ratio = v / total
        if ratio > threshold:
            results.append(f"{v}\t{ratio * 100.0:5.2f}\t{provider}\t{k}")

    return results


def no_loops_in_tf2(onnx_model):
    return not is_tf2 or all(n.op_type != "Loop" for n in onnx_model.graph.node)


def all_recurrents_should_bidirectional(onnx_model):
    return all([
        helper.get_attribute_value(attr) == b'bidirectional'
        for node in onnx_model.graph.node if node.op_type in ['GRU', 'LSTM', 'RNN']
        for attr in node.attribute if attr.name == 'direction'
    ])


def run_onnx_runtime(case_name, onnx_model, data, expected, model_files, rtol=1.e-3, atol=1.e-6,
                     compare_perf=False, enable_profiling=False):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
    onnx.save_model(onnx_model, temp_model_file)
    try:
        import onnxruntime
        if enable_profiling:
            from onnxruntime import SessionOptions
            sess_options = SessionOptions()
            sess_options.enable_profiling = True
            sess = onnxruntime.InferenceSession(temp_model_file, sess_options)
        else:
            sess = onnxruntime.InferenceSession(temp_model_file)
    except ImportError:
        mock_keras2onnx.common.k2o_logger().warning("Cannot import ONNXRuntime!")
        return True

    if isinstance(data, dict):
        feed_input = data
    else:
        data = data if isinstance(data, list) else [data]
        input_names = sess.get_inputs()
        # to avoid too complicated test code, we restrict the input name in Keras test cases must be
        # in alphabetical order. It's always true unless there is any trick preventing that.
        feed = zip(sorted(i_.name for i_ in input_names), data)
        feed_input = dict(feed)
    actual = sess.run(None, feed_input)
    if compare_perf:
        count = 10
        time_start = time.time()
        for i in range(count):
            sess.run(None, feed_input)
        time_end = time.time()
        print('avg ort time =' + str((time_end - time_start)/count))

    if enable_profiling:
        profile_file = sess.end_profiling()
        profile_records = load_profile_json(profile_file)
        lines = parse_profile_results(profile_records)
        print("Results:")
        print("-" * 64)
        for line in lines:
            print(line)

    if expected is None:
        return

    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]

    res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in range(len(expected)))

    if res and temp_model_file not in model_files:  # still keep the failed case files for the diagnosis.
        model_files.append(temp_model_file)

    if not res:
        for n_ in range(len(expected)):
            expected_list = expected[n_].flatten()
            actual_list = actual[n_].flatten()
            print_mismatches(case_name, n_, expected_list, actual_list, rtol, atol)

    return res


def run_keras_and_ort(case_name, onnx_model, keras_model, data, expected, model_files, rtol=1.e-3, atol=1.e-6,
                      compare_perf=False, enable_profiling=False):
    if compare_perf:
        count = 10
        time_start = time.time()
        for i in range(count):
            keras_model.predict(data)
        time_end = time.time()
        print('avg keras time =' + str((time_end - time_start) / count))
    return run_onnx_runtime(case_name, onnx_model, data, expected, model_files,
                            rtol=rtol, atol=atol, compare_perf=compare_perf, enable_profiling=enable_profiling)


def run_image(model, model_files, img_path, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, color_mode="rgb",
              target_size=224, compare_perf=False):
    if is_tf2:
        preprocess_input = keras.applications.imagenet_utils.preprocess_input
    else:
        preprocess_input = keras.applications.resnet50.preprocess_input
    image = keras.preprocessing.image

    try:
        if not isinstance(target_size, tuple):
            target_size = (target_size, target_size)
        if is_keras_older_than("2.2.3"):
            # color_mode is not supported in old keras version
            img = image.load_img(img_path, target_size=target_size)
        else:
            img = image.load_img(img_path, color_mode=color_mode, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if color_mode == "rgb":
            x = preprocess_input(x)
    except FileNotFoundError:
        return False, 'The image data does not exist.'

    msg = ''
    preds = None
    try:
        preds = model.predict(x)
        if compare_perf:
            count = 10
            time_start = time.time()
            for i in range(count):
                model.predict(x)
            time_end = time.time()
            print('avg keras time =' + str((time_end - time_start) / count))
    except RuntimeError:
        msg = 'keras prediction throws an exception for model ' + model.name + ', skip comparison.'

    onnx_model = mock_keras2onnx.convert_keras(model, model.name, target_opset=get_max_opset_supported_for_test())
    res = run_onnx_runtime(model_name, onnx_model, x, preds, model_files, rtol=rtol, atol=atol, compare_perf=compare_perf)
    return res, msg


def is_bloburl_access(url):
    try:
        response = urllib.request.urlopen(url)
        return response.getcode() == 200
    except urllib.error.URLError:
        return False


def get_max_opset_supported_by_ort():
    try:
        import onnxruntime as ort
        ort_ver = Version(ort.__version__)
        ort_ver = Version("{}.{}.0".format(ort_ver.major, ort_ver.minor)).base_version

        if ort_ver in ORT_OPSET_VERSION.keys():
            return ORT_OPSET_VERSION[ort_ver]
        else:
            print("Given onnxruntime version doesn't exist in ORT_OPSET_VERSION: {}".format(ort_ver))
            return None
    except ImportError:
        return None


def get_max_opset_supported_for_test():
    return min(get_max_opset_supported_by_ort(), get_maximum_opset_supported())


def convert_keras_for_test(model, name=None, target_opset=None, **kwargs):
    if target_opset is None:
        target_opset = get_max_opset_supported_by_ort()

    print("Trying to run test with opset version: {}".format(target_opset))
    return convert_keras(model=model, name=name, target_opset=target_opset, **kwargs)
