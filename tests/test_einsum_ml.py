# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for einsum decomposition."""

import unittest
from itertools import permutations
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, TensorProto, numpy_helper
from tf2onnx.optimizer.einsum_optimizer import (
    OnnxMicroRuntime,
    predict_transposition_cost,
    compute_transposition_features)
from tf2onnx import constants
from backend_test_base import Tf2OnnxBackendTestBase


class TestEinsumMl(Tf2OnnxBackendTestBase):
    "unit tests for einsum optimizer"

    def test_onnx_micro_runtime(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5, 5, 4]).astype(
            np.float32).reshape((3, 2))

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                nodes=[
                    helper.make_node('Add', ["X", "X"], ["temp"]),
                    helper.make_node('Add', ["X", "temp"], ["Y"]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertIn('X', out)
        self.assertIn('Y', out)
        self.assertIn('temp', out)
        self.assertEqual(len(out), 3)

    def test_onnx_micro_runtime_exc1(self):
        "test OnnxMicroRuntime"
        with self.assertRaises(TypeError):
            OnnxMicroRuntime(None)

    def test_onnx_micro_runtime_exc2(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5, 5, 4]).astype(
            np.float32).reshape((3, 2))

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                initializer=[
                    numpy_helper.from_array(np.array([1], dtype=np.float32), name="C1"),
                    numpy_helper.from_array(np.array([2], dtype=np.float32), name="C2"),
                ],
                nodes=[
                    helper.make_node('Add', ["X", "C1"], ["temp"]),
                    helper.make_node('Pow', ["temp", "C2"], ["Y"]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        with self.assertRaises(NotImplementedError):
            rt.run({'X': x})
        with self.assertRaises(TypeError):
            rt.run(x)

    def test_onnx_micro_runtime_shape(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5, 5, 4]).astype(
            np.float32).reshape((3, 2))

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
                nodes=[
                    helper.make_node('Shape', ["X"], ["Y"]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        assert_almost_equal(np.array(x.shape, dtype=np.int64), out['Y'])

    def test_onnx_micro_runtime_unsqueeze(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5, 5, 4]).astype(
            np.float32).reshape((3, 2))
        i = np.array([1]).astype(np.int64)

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None),
                        helper.make_tensor_value_info('I', TensorProto.INT64, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
                nodes=[
                    helper.make_node('Unsqueeze', ["X", "I"], ["Y"]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x, 'I': i})
        assert_almost_equal(np.array(x.reshape((3, 1, 2))), out['Y'])

    def test_onnx_micro_runtime_transpose(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5, 5, 4]).astype(
            np.float32).reshape((3, 2))

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                nodes=[
                    helper.make_node('Transpose', ["X"], ["Y"], perm=[1, 0]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        assert_almost_equal(x.T, out['Y'])

    def test_onnx_micro_runtime_matmul(self):
        "test OnnxMicroRuntime"
        opset = self.config.opset
        x = np.array([1, 2, 4, 5]).astype(
            np.float32).reshape((2, 2))

        model_def = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=constants.OPSET_TO_IR_VERSION[opset],
            producer_name='tf2onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, None)],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
                nodes=[
                    helper.make_node('MatMul', ["X", "X"], ["Y"]),
                ]))

        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        assert_almost_equal(np.matmul(x, x), out['Y'])

    def test_features(self):
        res = compute_transposition_features((3, 5, 7), (0, 1, 2))
        self.assertIsInstance(res, dict)
        self.assertEqual(res["edit"], 0)
        self.assertEqual(res["rot"], -1)
        res = compute_transposition_features((3, 5, 7), (2, 1, 0))
        self.assertEqual(res["edit"], 2)
        self.assertEqual(res["rot"], 0)
        self.assertEqual(res["rev"], 1)

    def test_cost(self):
        res = predict_transposition_cost((300, 500, 700), (0, 1, 2))
        self.assertIsInstance(res, float)
        self.assertGreater(res, 0)
        for shape in [(3, 5, 7), (30, 50, 70)]:
            for perm in permutations([0, 1, 2]):
                p = tuple(perm)
                cost = predict_transposition_cost(shape, p)
                if p[-1] == 2:
                    self.assertEqual(cost, 0)


if __name__ == "__main__":
    unittest.main()
