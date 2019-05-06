# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" test common utilities."""

import argparse
import os
import sys
import unittest
from collections import defaultdict

from distutils.version import LooseVersion
from parameterized import parameterized
import numpy as np
from tf2onnx import constants, logging, utils

__all__ = [
    "TestConfig",
    "get_test_config",
    "unittest_main",
    "check_onnxruntime_backend",
    "check_tf_min_version",
    "check_tf_max_version",
    "skip_tf_versions",
    "check_onnxruntime_min_version",
    "check_opset_min_version",
    "check_opset_max_version",
    "check_target",
    "skip_caffe2_backend",
    "skip_onnxruntime_backend",
    "skip_opset",
    "check_onnxruntime_incompatibility",
    "validate_const_node",
    "group_nodes_by_type",
    "test_ms_domain",
    "check_node_domain",
    "check_op_count"
]


# pylint: disable=missing-docstring

class TestConfig(object):
    def __init__(self):
        self.platform = sys.platform
        self.tf_version = utils.get_tf_version()
        self.opset = int(os.environ.get("TF2ONNX_TEST_OPSET", constants.PREFERRED_OPSET))
        self.target = os.environ.get("TF2ONNX_TEST_TARGET", ",".join(constants.DEFAULT_TARGET)).split(',')
        self.backend = os.environ.get("TF2ONNX_TEST_BACKEND", "onnxruntime")
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
            version = LooseVersion(version)
        return version

    def __str__(self):
        return "\n\t".join(["TestConfig:",
                            "platform={}".format(self.platform),
                            "tf_version={}".format(self.tf_version),
                            "opset={}".format(self.opset),
                            "target={}".format(self.target),
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


def check_tf_max_version(max_accepted_version, message=""):
    """ Skip if tf_version > max_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tf <= {}".format(max_accepted_version), message)
    return unittest.skipIf(config.tf_version > LooseVersion(max_accepted_version), reason)


def check_tf_min_version(min_required_version, message=""):
    """ Skip if tf_version < min_required_version """
    config = get_test_config()
    reason = _append_message("conversion requires tf >= {}".format(min_required_version), message)
    return unittest.skipIf(config.tf_version < LooseVersion(min_required_version), reason)


def skip_tf_versions(excluded_versions, message=""):
    """ Skip if tf_version SEMANTICALLY matches any of excluded_versions. """
    config = get_test_config()
    condition = False
    reason = _append_message("conversion excludes tf {}".format(excluded_versions), message)

    current_tokens = str(config.tf_version).split('.')
    for excluded_version in excluded_versions:
        exclude_tokens = excluded_version.split('.')
        # assume len(exclude_tokens) <= len(current_tokens)
        for i, exclude in enumerate(exclude_tokens):
            if not current_tokens[i] == exclude:
                break
        condition = True

    return unittest.skipIf(condition, reason)


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
                           config.backend_version < LooseVersion(min_required_version), reason)


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
        res[node.type].append(node)
    return res


def check_op_count(graph, op_type, expected_count):
    return len(group_nodes_by_type(graph)[op_type]) == expected_count


def check_lstm_count(graph, expected_count):
    return check_op_count(graph, "LSTM", expected_count)


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
