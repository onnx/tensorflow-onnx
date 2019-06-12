# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for tf.cond and tf.case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from onnx import helper, TensorProto
from tf2onnx import utils
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ


class UtilTests(Tf2OnnxBackendTestBase):
    def test_tensor_equal_bad_type(self):
        t1 = helper.make_tensor(name='tensor_1',
                           data_type=TensorProto.INT64,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int64))
        t2 = helper.make_tensor(name='tensor_2',
                           data_type=TensorProto.INT32,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int32))
        self.assertFalse(utils.are_tensors_equal(t1, t2))

    def test_tensor_equal_bad_shape(self):
        t1 = helper.make_tensor(name='tensor_1',
                           data_type=TensorProto.INT64,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int64))
        t2 = helper.make_tensor(name='tensor_2',
                           data_type=TensorProto.INT64,
                           dims=[2,3,1],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int64))
        self.assertFalse(utils.are_tensors_equal(t1, t2))

    def test_tensor_equal_int32(self):
        t1 = helper.make_tensor(name='tensor_1',
                           data_type=TensorProto.INT32,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int32))
        t2 = helper.make_tensor(name='tensor_2',
                           data_type=TensorProto.INT32,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int32))
        self.assertTrue(utils.are_tensors_equal(t1, t2))

    def test_tensor_equal_int64(self):
        t1 = helper.make_tensor(name='tensor_1',
                           data_type=TensorProto.INT64,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int64))
        t2 = helper.make_tensor(name='tensor_2',
                           data_type=TensorProto.INT64,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int64))
        self.assertTrue(utils.are_tensors_equal(t1, t2))

    def test_tensor_equal_float(self):
        t1 = helper.make_tensor(name='tensor_1',
                                data_type=TensorProto.FLOAT,
                                dims=[2, 3],
                                vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.float32))
        t2 = helper.make_tensor(name='tensor_2',
                                data_type=TensorProto.FLOAT,
                                dims=[2, 3],
                                vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.float32))
        self.assertTrue(utils.are_tensors_equal(t1, t2))

    def test_tensor_equal_double(self):
        t1 = helper.make_tensor(name='tensor_1',
                                data_type=TensorProto.DOUBLE,
                                dims=[2, 3],
                                vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.float64))
        t2 = helper.make_tensor(name='tensor_2',
                                data_type=TensorProto.DOUBLE,
                                dims=[2, 3],
                                vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.float64))
        self.assertTrue(utils.are_tensors_equal(t1, t2))

    def test_graph_equal(self):
        const_1_val = [1.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        node1 = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")
        node2 = helper.make_node("Add", ["const_1", 'X'], ["Add"], name="add_1")

        graph1 = helper.make_graph(
            [node1, node2],
            "graph-equal-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        graph2 = helper.make_graph(
            [node1, node2],
            "graph-equal-test-2",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        self.assertTrue(utils.are_graphs_equal(graph1, graph2))

    def test_graph_not_equal(self):
        const_1_val = [1.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        node1 = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")
        node2 = helper.make_node("Add", ["const_1", 'X'], ["Add"], name="add_1")

        graph1 = helper.make_graph(
            [node1, node2],
            "graph-equal-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        graph2 = helper.make_graph(
            [node1, node2],
            "graph-equal-test-2",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 1))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        self.assertFalse(utils.are_graphs_equal(graph1, graph2))

    def test_graphs_equal(self):
        const_1_val = [1.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        node1 = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")
        node2 = helper.make_node("Add", ["const_1", 'X'], ["Add"], name="add_1")
        identity1 = helper.make_node("Identity", ["const_1"], ["id_1"], name="id_1")

        graph1 = helper.make_graph(
            [node1, node2],
            "graph-equal-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        graph2 = helper.make_graph(
            [node1, node2],
            "graph-equal-test-2",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        graph3 = helper.make_graph(
            [identity1],
            "graph-equal-test-3",
            [],
            [helper.make_tensor_value_info("id_1", TensorProto.FLOAT, (2, 3))],
        )
        graph4 = helper.make_graph(
            [identity1],
            "graph-equal-test-4",
            [],
            [helper.make_tensor_value_info("id_1", TensorProto.FLOAT, (2, 3))],
        )
        self.assertTrue(utils.are_graph_lists_equal([graph1,graph3], [graph2, graph4]))

    def test_attr_equal(self):
        const_1_val = [1.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        node1 = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")
        node2 = helper.make_node("Add", ["const_1", 'X'], ["Add"], name="add_1")

        graph1 = helper.make_graph(
            [node1, node2],
            "graph-equal-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        graph2 = helper.make_graph(
            [node1, node2],
            "graph-equal-test-2",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Add", TensorProto.FLOAT, (2, 3))],
        )
        t1 = helper.make_tensor(name='tensor_1',
                           data_type=TensorProto.INT32,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int32))
        t2 = helper.make_tensor(name='tensor_2',
                           data_type=TensorProto.INT32,
                           dims=[2,3],
                           vals=np.array([1, 2, 3, 4, 5, 6]).astype(np.int32))
        attr1 = {
            'int_val': helper.make_attribute('int_val', 1),
            'float_val': helper.make_attribute('float_val', 2.1),
            'string_val': helper.make_attribute('string_val', 'some string'),
            'tensor_val': helper.make_attribute('tensor_val', t1),
            'graph_val': helper.make_attribute('graph_val', graph1)
        }
        attr2 = {
            'int_val': helper.make_attribute('int_val', 1),
            'float_val': helper.make_attribute('float_val', 2.1),
            'string_val': helper.make_attribute('string_val', 'some string'),
            'tensor_val': helper.make_attribute('tensor_val', t2),
            'graph_val': helper.make_attribute('graph_val', graph2)
        }
        self.assertTrue(utils.are_attr_equal(attr1, attr2))


if __name__ == '__main__':
    unittest_main()
