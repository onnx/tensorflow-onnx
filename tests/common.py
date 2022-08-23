# SPDX-License-Identifier: Apache-2.0


""" test common utilities."""

import argparse
import os
import sys
import unittest
from collections import defaultdict

from packaging.version import Version
from parameterized import parameterized
import numpy as np
import tensorflow as tf

from tf2onnx import constants, logging, utils, tf_utils, tf_loader

# pylint: disable=import-outside-toplevel
__all__ = [
    "TestConfig",
    "get_test_config",
    "unittest_main",
    "check_onnxruntime_backend",
    "check_tf_min_version",
    "check_tf_max_version",
    "check_tfjs_min_version",
    "check_tfjs_max_version",
    "skip_tf_versions",
    "skip_tf_cpu",
    "check_onnxruntime_min_version",
    "check_opset_min_version",
    "check_opset_max_version",
    "skip_tf2",
    "skip_tflite",
    "skip_tfjs",
    "requires_tflite",
    "check_opset_after_tf_version",
    "check_target",
    "skip_caffe2_backend",
    "allow_missing_shapes",
    "skip_onnx_checker",
    "skip_onnxruntime_backend",
    "skip_opset",
    "check_onnxruntime_incompatibility",
    "validate_const_node",
    "group_nodes_by_type",
    "test_ms_domain",
    "check_node_domain",
    "check_op_count",
    "check_gru_count",
    "check_lstm_count",
]


# pylint: disable=missing-docstring,unused-argument

class TestConfig(object):
    def __init__(self):
        self.platform = sys.platform
        self.tf_version = tf_utils.get_tf_version()
        self.opset = int(os.environ.get("TF2ONNX_TEST_OPSET", constants.PREFERRED_OPSET))
        self.target = os.environ.get("TF2ONNX_TEST_TARGET", ",".join(constants.DEFAULT_TARGET)).split(',')
        self.backend = os.environ.get("TF2ONNX_TEST_BACKEND", "onnxruntime")
        self.skip_tflite_tests = os.environ.get("TF2ONNX_SKIP_TFLITE_TESTS", "FALSE").upper() == "TRUE"
        self.skip_tfjs_tests = os.environ.get("TF2ONNX_SKIP_TFJS_TESTS", "FALSE").upper() == "TRUE"
        self.skip_tf_tests = os.environ.get("TF2ONNX_SKIP_TF_TESTS", "FALSE").upper() == "TRUE"
        self.skip_onnx_checker = False
        self.allow_missing_shapes = False
        self.run_tfl_consistency_test = os.environ.get("TF2ONNX_RUN_TFL_CONSISTENCY_TEST", "FALSE").upper() == "TRUE"
        self.backend_version = self._get_backend_version()
        self.log_level = logging.WARNING
        self.temp_dir = utils.get_temp_directory()

    @property
    def is_mac(self):
        return self.platform == "darwin"

    @property
    def is_onnxruntime_backend(self):
        return self.backend == "onnxruntime"

    @property
    def is_caffe2_backend(self):
        return self.backend == "caffe2"

    @property
    def is_debug_mode(self):
        return utils.is_debug_mode()

    def _get_backend_version(self):
        version = None
        if self.backend == "onnxruntime":
            import onnxruntime as ort
            version = ort.__version__
        elif self.backend == "caffe2":
            # TODO: get caffe2 version
            pass

        if version:
            version = Version(version)
        return version

    def __str__(self):
        return "\n\t".join(["TestConfig:",
                            "platform={}".format(self.platform),
                            "tf_version={}".format(self.tf_version),
                            "opset={}".format(self.opset),
                            "target={}".format(self.target),
                            "skip_tflite_tests={}".format(self.skip_tflite_tests),
                            "skip_tfjs_tests={}".format(self.skip_tfjs_tests),
                            "skip_tf_tests={}".format(self.skip_tf_tests),
                            "run_tfl_consistency_test={}".format(self.run_tfl_consistency_test),
                            "backend={}".format(self.backend),
                            "backend_version={}".format(self.backend_version),
                            "is_debug_mode={}".format(self.is_debug_mode),
                            "temp_dir={}".format(self.temp_dir)])

    @staticmethod
    def load():
        config = TestConfig()
        # if not launched by pytest, parse console arguments to override config
        if "pytest" not in sys.argv[0]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--backend", default=config.backend,
                                choices=["caffe2", "onnxruntime"],
                                help="backend to test against")
            parser.add_argument("--opset", type=int, default=config.opset, help="opset to test against")
            parser.add_argument("--target", default=",".join(config.target), choices=constants.POSSIBLE_TARGETS,
                                help="target platform")
            parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
            parser.add_argument("--debug", help="output debugging information", action="store_true")
            parser.add_argument("--temp_dir", help="temp dir")
            parser.add_argument("unittest_args", nargs='*')

            args = parser.parse_args()
            if args.debug:
                utils.set_debug_mode(True)

            config.backend = args.backend
            config.opset = args.opset
            config.target = args.target.split(',')
            config.log_level = logging.get_verbosity_level(args.verbose, config.log_level)
            if args.temp_dir:
                config.temp_dir = args.temp_dir

            # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
            sys.argv[1:] = args.unittest_args

        return config


# need to load config BEFORE main is executed when launched from script
# otherwise, it will be too late for test filters to take effect
_config = TestConfig.load()


def get_test_config():
    global _config
    return _config


def unittest_main():
    config = get_test_config()
    logging.basicConfig(level=config.log_level)
    with logging.set_scope_level(logging.INFO) as logger:
        logger.info(config)
    unittest.main()


def _append_message(reason, message):
    if message:
        reason = reason + ": " + message
    return reason


def check_opset_after_tf_version(tf_version, required_opset, message=""):
    """ Skip if tf_version > max_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires opset {} after tf {}".format(required_opset, tf_version), message)
    skip = config.tf_version >= Version(tf_version) and config.opset < required_opset
    return unittest.skipIf(skip, reason)


def skip_tf2(message=""):
    """ Skip if tf_version > max_required_version """
    reason = _append_message("test needs to be fixed for tf-2.x", message)
    return unittest.skipIf(tf_loader.is_tf2(), reason)


def skip_tfjs(message=""):
    """ Skip the tfjs conversion for this test """
    config = get_test_config()
    reason = _append_message("test disabled for tfjs", message)
    if config.skip_tf_tests and config.skip_tflite_tests:
        # If we are skipping tf and tflite also, there is no reason to run this test
        return unittest.skip(reason)
    def decorator(func):
        def test(self):
            tmp = config.skip_tfjs_tests
            config.skip_tfjs_tests = True
            try:
                func(self)
            finally:
                config.skip_tfjs_tests = tmp
        return test
    return decorator


def skip_tflite(message=""):
    """ Skip the tflite conversion for this test """
    config = get_test_config()
    reason = _append_message("test disabled for tflite", message)
    if config.skip_tf_tests and config.skip_tfjs_tests:
        # If we are skipping tf and tfjs also, there is no reason to run this test
        return unittest.skip(reason)
    def decorator(func):
        def test(self):
            tmp = config.skip_tflite_tests
            config.skip_tflite_tests = True
            try:
                func(self)
            finally:
                config.skip_tflite_tests = tmp
        return test
    return decorator


def skip_onnx_checker(message=""):
    """ Skip running the onnx checker for this test """
    config = get_test_config()
    def decorator(func):
        def test(self):
            tmp = config.skip_onnx_checker
            config.skip_onnx_checker = True
            try:
                func(self)
            finally:
                config.skip_onnx_checker = tmp
        return test
    return decorator


def allow_missing_shapes(message=""):
    """ Only check for incompatible, not missing shapes/dims """
    config = get_test_config()
    def decorator(func):
        def test(self):
            tmp = config.allow_missing_shapes
            config.allow_missing_shapes = True
            try:
                func(self)
            finally:
                config.allow_missing_shapes = tmp
        return test
    return decorator



def requires_tflite(message=""):
    """ Skip test if tflite tests are disabled """
    config = get_test_config()
    reason = _append_message("test requires tflite", message)
    return unittest.skipIf(config.skip_tflite_tests, reason)


def requires_custom_ops(message=""):
    """ Skip until custom ops framework is on PyPI. """
    reason = _append_message("test needs custom ops framework", message)
    try:
        import onnxruntime_extensions  #pylint: disable=import-outside-toplevel,unused-import
        can_import = True
    except ModuleNotFoundError:
        can_import = False
    return unittest.skipIf(not can_import, reason)

def check_tfjs_max_version(max_accepted_version, message=""):
    """ Skip if tfjs_version > max_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tensorflowjs <= {}".format(max_accepted_version), message)
    try:
        import tensorflowjs
        can_import = True
    except ModuleNotFoundError:
        can_import = False
    return unittest.skipIf(can_import and not config.skip_tfjs_tests and \
         Version(tensorflowjs.__version__) > Version(max_accepted_version), reason)

def check_tfjs_min_version(min_required_version, message=""):
    """ Skip if tjs_version < min_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tensorflowjs >= {}".format(min_required_version), message)
    try:
        import tensorflowjs
        can_import = True
    except ModuleNotFoundError:
        can_import = False
    return unittest.skipIf(can_import and not config.skip_tfjs_tests and \
         Version(tensorflowjs.__version__) < Version(min_required_version), reason)

def check_tf_max_version(max_accepted_version, message=""):
    """ Skip if tf_version > max_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tf <= {}".format(max_accepted_version), message)
    return unittest.skipIf(config.tf_version > Version(max_accepted_version), reason)


def check_tf_min_version(min_required_version, message=""):
    """ Skip if tf_version < min_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tf >= {}".format(min_required_version), message)
    return unittest.skipIf(config.tf_version < Version(min_required_version), reason)


def skip_tf_versions(excluded_versions, message=""):
    """ Skip if tf_version matches any of excluded_versions. """
    if not isinstance(excluded_versions, list):
        excluded_versions = [excluded_versions]
    config = get_test_config()
    condition = False
    reason = _append_message("conversion excludes tf {}".format(excluded_versions), message)

    for excluded_version in excluded_versions:
        # tf version with same specificity as excluded_version
        tf_version = '.'.join(str(config.tf_version).split('.')[:excluded_version.count('.') + 1])
        if excluded_version == tf_version:
            condition = True

    return unittest.skipIf(condition, reason)


def is_tf_gpu():
    return tf.test.is_gpu_available()


def skip_tf_cpu(message=""):
    is_tf_cpu = not is_tf_gpu()
    return unittest.skipIf(is_tf_cpu, message)


def check_opset_min_version(min_required_version, message=""):
    """ Skip if opset < min_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires opset >= {}".format(min_required_version), message)
    return unittest.skipIf(config.opset < min_required_version, reason)


def check_opset_max_version(max_accepted_version, message=""):
    """ Skip if opset > max_accepted_version """
    config = get_test_config()
    reason = _append_message("conversion requires opset <= {}".format(max_accepted_version), message)
    return unittest.skipIf(config.opset > max_accepted_version, reason)


def skip_opset(opset_v, message=""):
    """ Skip if opset = opset_v """
    config = get_test_config()
    reason = _append_message("conversion requires opset != {}".format(opset_v), message)
    return unittest.skipIf(config.opset == opset_v, reason)


def check_target(required_target, message=""):
    """ Skip if required_target is NOT specified """
    config = get_test_config()
    reason = _append_message("conversion requires target {} specified".format(required_target), message)
    return unittest.skipIf(required_target not in config.target, reason)


def skip_onnxruntime_backend(message=""):
    """ Skip if backend is onnxruntime """
    config = get_test_config()
    reason = _append_message("not supported by onnxruntime", message)
    return unittest.skipIf(config.is_onnxruntime_backend, reason)


def check_onnxruntime_backend(message=""):
    """ Skip if backend is NOT onnxruntime """
    config = get_test_config()
    reason = _append_message("only supported by onnxruntime", message)
    return unittest.skipIf(not config.is_onnxruntime_backend, reason)


def check_onnxruntime_min_version(min_required_version, message=""):
    """ Skip if onnxruntime version < min_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires onnxruntime >= {}".format(min_required_version), message)
    return unittest.skipIf(config.is_onnxruntime_backend and
                           config.backend_version < Version(min_required_version), reason)


def skip_caffe2_backend(message=""):
    """ Skip if backend is caffe2 """
    config = get_test_config()
    reason = _append_message("not supported by caffe2", message)
    return unittest.skipIf(config.is_caffe2_backend, reason)


def check_onnxruntime_incompatibility(op):
    """ Skip if backend is onnxruntime AND op is NOT supported in current opset """
    config = get_test_config()

    if not config.is_onnxruntime_backend:
        return unittest.skipIf(False, None)

    support_since = {
        "Abs": 6,  # Abs-1
        "Add": 7,  # Add-1, Add-6
        "AveragePool": 7,  # AveragePool-1
        "Div": 7,  # Div-1, Div-6
        "Elu": 6,  # Elu-1
        "Equal": 7,  # Equal-1
        "Exp": 6,  # Exp-1
        "Greater": 7,  # Greater-1
        "Less": 7,  # Less-1
        "Log": 6,  # Log-1
        "Max": 6,  # Max-1
        "Min": 6,  # Min-1
        "Mul": 7,  # Mul-1, Mul-6
        "Neg": 6,  # Neg-1
        "Pow": 7,  # Pow-1
        "Reciprocal": 6,  # Reciprocal-1
        "Relu": 6,  # Relu-1
        "Sqrt": 6,  # Sqrt-1
        "Sub": 7,  # Sub-1, Sub-6
        "Tanh": 6,  # Tanh-1
    }

    if op not in support_since or config.opset >= support_since[op]:
        return unittest.skipIf(False, None)

    reason = "{} is not supported by onnxruntime before opset {}".format(op, support_since[op])
    return unittest.skipIf(True, reason)


def validate_const_node(node, expected_val):
    if node.is_const():
        node_val = node.get_tensor_value()
        np.testing.assert_allclose(expected_val, node_val)
        return True
    return False


def group_nodes_by_type(graph):
    res = defaultdict(list)
    for node in graph.get_nodes():
        attr_body_graphs = node.get_body_graphs()
        if attr_body_graphs:
            for _, body_graph in attr_body_graphs.items():
                body_graph_res = group_nodes_by_type(body_graph)
                for k, v in body_graph_res.items():
                    res[k].extend(v)
        res[node.type].append(node)
    return res


def check_op_count(graph, op_type, expected_count, disabled=True):
    # The grappler optimization may change some of the op counts.
    return disabled or len(group_nodes_by_type(graph)[op_type]) == expected_count


def check_lstm_count(graph, expected_count):
    return len(group_nodes_by_type(graph)["LSTM"]) == expected_count


def check_gru_count(graph, expected_count):
    return check_op_count(graph, "GRU", expected_count)


_MAX_MS_OPSET_VERSION = 1


def test_ms_domain(versions=None):
    """ Parameterize test case to apply ms opset(s) as extra_opset. """

    @check_onnxruntime_backend()
    def _custom_name_func(testcase_func, param_num, param):
        del param_num
        arg = param.args[0]
        return "%s_%s" % (testcase_func.__name__, arg.version)

    # Test all opset versions in ms domain if versions is not specified
    if versions is None:
        versions = list(range(1, _MAX_MS_OPSET_VERSION + 1))

    opsets = []
    for version in versions:
        opsets.append([utils.make_opsetid(constants.MICROSOFT_DOMAIN, version)])
    return parameterized.expand(opsets, testcase_func_name=_custom_name_func)


def check_node_domain(node, domain):
    # None or empty string means onnx domain
    if not domain:
        return not node.domain
    return node.domain == domain
