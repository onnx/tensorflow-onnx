# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for gru."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tf2onnx.tf_loader import is_tf2

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,cell-var-from-loop

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFINPUT3 = "input3"
_INPUT3 = "input3:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"
_TFOUTPUT2 = "output2"
_OUTPUT2 = "output2:0"
_TFOUTPUT3 = "output3"
_OUTPUT3 = "output3:0"

if is_tf2():
    # There is no LSTMBlockCell in tf-2.x
    BasicLSTMCell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell
    LSTMCell = tf.compat.v1.nn.rnn_cell.LSTMCell
    GRUCell = tf.compat.v1.nn.rnn_cell.GRUCell
    MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.compat.v1.nn.bidirectional_dynamic_rnn
else:
    BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
    LSTMCell = tf.contrib.rnn.LSTMCell
    GRUCell = tf.contrib.rnn.GRUCell
    LSTMBlockCell = tf.contrib.rnn.LSTMBlockCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    dynamic_rnn = tf.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn


# TODO: as a workaround, set batch_size to 1 for now to bypass a onnxruntime bug, revert it when the bug is fixed
class GRUTests(Tf2OnnxBackendTestBase):

    def run_test_case(self, *args, graph_validator=None, **kwargs):  #pylint: disable=arguments-differ
        # TF GRU has an unknown dim
        tmp = self.config.allow_missing_shapes
        self.config.allow_missing_shapes = True
        def new_graph_validator(g):
            good = True
            if graph_validator is not None:
                good = good and graph_validator(g)
            if is_tf2() and ':' in g.outputs[0]:
                # Only check for tf2 and tfjs, not tflite
                good = good and check_op_count(g, "Loop", 0, disabled=False)
                good = good and check_op_count(g, "Scan", 0, disabled=False)
            return good
        try:
            super().run_test_case(*args, graph_validator=new_graph_validator, **kwargs)
        finally:
            self.config.allow_missing_shapes = tmp

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # no scope
            cell = GRUCell(
                units,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_multiple_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            gru_output_list = []
            gru_cell_state_list = []
            # no scope
            cell = GRUCell(
                units,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            gru_output_list.append(outputs)
            gru_cell_state_list.append(cell_state)

            # given scope
            cell = GRUCell(
                units,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=[4],
                    scope=scope)
            gru_output_list.append(outputs)
            gru_cell_state_list.append(cell_state)

            return tf.identity(gru_output_list, name="output"), tf.identity(gru_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06)
        # graph_validator=lambda g: check_gru_count(g, 2))

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_single_dynamic_gru_seq_length_is_const(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = GRUCell(
                units,
                kernel_initializer=initializer)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=[5])

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_single_dynamic_gru_seq_length_is_not_const(self):
        for np_dtype in [np.int32, np.int64, np.float32]:
            units = 5
            batch_size = 1
            x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
            x_val = np.stack([x_val] * batch_size)
            y_val = np.array([5], dtype=np_dtype)

            def func(x, seq_length):
                initializer = init_ops.constant_initializer(0.5)

                # no scope
                cell = GRUCell(
                    units,
                    kernel_initializer=initializer)
                outputs, cell_state = dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=tf.identity(seq_length))

                return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

            feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
            input_names_with_port = ["input_1:0", "input_2:0"]
            output_names_with_port = ["output:0", "cell_state:0"]
            self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                               graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_gru_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 1)
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = GRUCell(
                units,
                kernel_initializer=initializer)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)  # by default zero initializer is used

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_gru_ch_zero_state_initializer(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = GRUCell(
                units,
                kernel_initializer=initializer)

            # defining initial state
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                initial_state=initial_state,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_gru_random_weights(self):
        hidden_size = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=42)

            # no scope
            cell = GRUCell(
                hidden_size,
                kernel_initializer=initializer)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_gru_random_weights2(self):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)


        def func(x):
            #initializer = tf.constant_initializer(5.0)
            initializer = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            # no scope
            cell = GRUCell(
                hidden_size,
                kernel_initializer=initializer)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.01,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_dynamic_gru_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=42)
            cell1 = GRUCell(
                units,
                kernel_initializer=initializer)

            outputs, _ = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_dynamic_gru_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=42)
            cell1 = GRUCell(
                units,
                kernel_initializer=initializer)

            _, cell_state = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bigru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bigru, no scope
            cell1 = GRUCell(
                units,
                kernel_initializer=initializer)
            cell2 = GRUCell(
                units,
                kernel_initializer=initializer)
            outputs, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bigru_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bigru, no scope
            cell1 = GRUCell(
                units,
                kernel_initializer=initializer)
            cell2 = GRUCell(
                units,
                kernel_initializer=initializer)
            outputs, _ = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bigru_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bigru, no scope
            cell1 = GRUCell(
                units,
                kernel_initializer=initializer)
            cell2 = GRUCell(
                units,
                kernel_initializer=initializer)
            _, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bidirectional_but_one_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bigru, no scope
            cell = GRUCell(
                units,
                kernel_initializer=initializer)
            outputs, cell_state = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bidirectional_but_one_gru_and_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):

            # bigru, no scope
            cell = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            outputs, _ = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bidirectional_but_one_gru_and_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):

            # bigru, no scope
            cell = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            _, cell_state = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bigru_unknown_batch_size(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):

            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            _, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32,
            )

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bigru_outputs_partially_consumed(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):

            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            (output_fw, _), (_, state_bw) = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(output_fw, name="output"), tf.identity(state_bw, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @skip_tf_versions(["2.1"], "TF fails to correctly add output_2 node.")
    def test_dynamic_multi_bigru_with_same_input_hidden_size(self):
        batch_size = 10
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bigru, no scope
            units = 5
            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            outputs_1, cell_state_1 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32,
                scope="bigru_1"
            )

            units = 10
            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            outputs_2, cell_state_2 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32,
                scope="bigru_2"
            )

            return tf.identity(outputs_1, name="output_1"),  \
                   tf.identity(cell_state_1, name="cell_state_1"), \
                   tf.identity(outputs_2, name="output_2"), \
                   tf.identity(cell_state_2, name="cell_state_2")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output_1:0", "cell_state_1:0", "output_2:0", "cell_state_2:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06)
        # graph_validator=lambda g: check_gru_count(g, 2))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @skip_tf_versions(["2.1"], "TF fails to correctly add output_2 node.")
    def test_dynamic_multi_bigru_with_same_input_seq_len(self):
        units = 5
        batch_size = 10
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        seq_len_val = np.array([3], dtype=np.int32)

        def func(x, y1, y2):
            seq_len1 = tf.tile(y1, [batch_size])
            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=45))
            outputs_1, cell_state_1 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                sequence_length=seq_len1,
                dtype=tf.float32,
                scope="bigru_1"
            )
            seq_len2 = tf.tile(y2, [batch_size])
            cell1 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=46),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=47))
            cell2 = GRUCell(
                units,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=48),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=49))
            outputs_2, cell_state_2 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                sequence_length=seq_len2,
                dtype=tf.float32,
                scope="bigru_2"
            )

            return tf.identity(outputs_1, name="output_1"), \
                   tf.identity(cell_state_1, name="cell_state_1"), \
                   tf.identity(outputs_2, name="output_2"), \
                   tf.identity(cell_state_2, name="cell_state_2")

        feed_dict = {"input_1:0": x_val, "input_2:0": seq_len_val, "input_3:0": seq_len_val}
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        output_names_with_port = ["output_1:0", "cell_state_1:0", "output_2:0", "cell_state_2:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06)
        # graph_validator=lambda g: check_gru_count(g, 2))

    @check_tf_min_version("2.2")
    def test_keras_gru(self):
        in_shape = [10, 3]
        x_val = np.random.uniform(size=[2, 10, 3]).astype(np.float32)

        model_in = tf.keras.layers.Input(tuple(in_shape), batch_size=2)
        x = tf.keras.layers.GRU(5, return_sequences=True, return_state=True,
                                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                                recurrent_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43))(model_in)
        model = tf.keras.models.Model(inputs=model_in, outputs=x)

        def func(x):
            y = model(x)
            return tf.identity(y[0], name=_TFOUTPUT), tf.identity(y[1], name=_TFOUTPUT1)
        self.run_test_case(func, {_INPUT: x_val}, [], [_OUTPUT, _OUTPUT1], rtol=1e-05, atol=1e-06)


if __name__ == '__main__':
    unittest_main()
