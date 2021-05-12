# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for optimizers such as TransposeOptimizer."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, numpy_helper, TensorProto, OperatorSetIdProto
from onnxruntime import InferenceSession
from parameterized import parameterized

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, group_nodes_by_type, check_opset_min_version, check_opset_max_version, get_test_config
from tf2onnx import utils, constants
from tf2onnx.graph import GraphUtil
from tf2onnx.optimizer import EinsumOptimizer


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class EinsumOptimizerTests(Tf2OnnxBackendTestBase):
    """Run original model proto and modified model proto with onnxruntime, compare the results."""

    def run_and_compare(self, output_names_with_port, onnx_feed_dict, origin_proto, op_type,
                        debug=False, rtol=1e-07):
        optimizers = OrderedDict([("optimize_einsum", EinsumOptimizer)])
        utils.make_sure(op_type is not None,
                        "op_type should be specified")
        utils.make_sure(self.config.is_onnxruntime_backend,
                        "only onnxruntime is supported to test transpose optimizer")

        origin_model_path = self.save_onnx_model(
            origin_proto, onnx_feed_dict, postfix="_origin")
        expected = self.run_onnxruntime(
            origin_model_path, onnx_feed_dict, output_names_with_port)

        new_proto, new_graph = GraphUtil.optimize_model_proto(
            origin_proto, catch_errors=False, return_graph=True, optimizers=optimizers)
        self.assertTrue(new_proto,
                        msg="model proto after optimizer should not be None")
                        
        new_model_path = self.save_onnx_model(new_proto, onnx_feed_dict, postfix="_opt")
        current = GraphUtil.get_node_count_from_onnx_graph(new_proto.graph)
        actual = self.run_onnxruntime(new_model_path, onnx_feed_dict, output_names_with_port)

        for expected_val, actual_val in zip(expected, actual):
            self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=1e-5)
            self.assertEqual(expected_val.dtype, actual_val.dtype)
            self.assertEqual(expected_val.shape, actual_val.shape)

        self.assert_shapes_correct(new_graph, allow_missing=False, run_checker=True)
        return new_proto

    def run_einsum_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                           remaining_transpose_num=None, debug=False, rtol=1e-07):
        return self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type="Einsum",
                                    debug=debug, rtol=rtol)

    def make_model(self, graph, producer_name="onnx-tests"):
        imp = OperatorSetIdProto()
        imp.version = self.config.opset
        model_proto = helper.make_model(graph, producer_name=producer_name, opset_imports=[imp])
        try:
            model_proto.ir_version = constants.OPSET_TO_IR_VERSION.get(
                self.config.opset, model_proto.ir_version)
        except:  # pylint: disable=bare-except
            pass
        return model_proto

    def common_einsum(self, equation, operands=None):
        if operands is not None:
            inputs = operands
        else:
            eqs = equation.split("->")[0].split(",")
            inputs = []
            for d, eq in enumerate(eqs):
                i = np.arange(2 ** len(eq)).reshape(
                    (2,) * len(eq)).astype(np.float32)
                inputs.append(
                    i + np.array([3 ** d], dtype=np.float32))
        output_dim = len(equation.split("->")[1])
        node1 = helper.make_node("Einsum", ["X0", "X1"], ["Y"], equation=equation)
        graph = helper.make_graph(
            [node1],
            "test_optimization",
            [helper.make_tensor_value_info(
                "X%d" % i, TensorProto.FLOAT, [None] * len(operands[i].shape))
             for i in range(len(operands))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None] * output_dim)])

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"X%d" % i: o for i, o in enumerate(operands)}
        new_model_proto = self.run_einsum_compare(["Y"], feed_dict, model_proto)

        sess1 = InferenceSession(model_proto.SerializeToString())
        sess2 = InferenceSession(new_model_proto.SerializeToString())
        got1 = sess1.run(None, feed_dict)
        got2 = sess2.run(None, feed_dict)
        assert_almost_equal(got1, got2)
        self.assertNotIn('Einsum', str(new_model_proto))        

    def test_np_test_broadcasting_dot_cases2(self):
        f = np.arange(7 * 55).reshape(7, 11, 5).astype(np.float32)
        g = np.arange(30).reshape(2, 3, 5).astype(np.float32)
        self.common_einsum('obk,ijk->ioj', operands=[f, g])


if __name__ == "__main__":
    unittest_main()
