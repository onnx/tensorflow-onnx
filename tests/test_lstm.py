# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for lstm."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import check_tf_min_version, unittest_main, check_opset_after_tf_version, \
    skip_tf2, skip_tf_versions, check_op_count, skip_tfjs

from tf2onnx.tf_loader import is_tf2


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,cell-var-from-loop
# pylint: disable=invalid-name


if is_tf2():
    # There is no LSTMBlockCell in tf-2.x
    BasicLSTMCell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell
    LSTMCell = tf.compat.v1.nn.rnn_cell.LSTMCell
    MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.compat.v1.nn.bidirectional_dynamic_rnn
else:
    BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
    LSTMCell = tf.contrib.rnn.LSTMCell
    LSTMBlockCell = tf.contrib.rnn.LSTMBlockCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    dynamic_rnn = tf.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

# pylint: enable=invalid-name

class LSTMTests(Tf2OnnxBackendTestBase):

    def run_test_case(self, *args, require_lstm_count=1,  #pylint: disable=arguments-differ
                      graph_validator=None, **kwargs):
        # TF LSTM has an unknown dim
        tmp = self.config.allow_missing_shapes
        self.config.allow_missing_shapes = True
        def new_graph_validator(g):
            good = True
            if graph_validator is not None:
                good = good and graph_validator(g)
            if require_lstm_count is None or ":" not in g.outputs[0]:
                # Skip checks for tflite graphs (no ":" in outputs)
                return good
            good = good and check_op_count(g, "LSTM", require_lstm_count, disabled=False)
            # If LSTM op rewriter failed to work, Loop op will be shown in general.
            good = good and check_op_count(g, "Loop", 0, disabled=False)
            return good
        try:
            super().run_test_case(*args, graph_validator=new_graph_validator, **kwargs)
        finally:
            self.config.allow_missing_shapes = tmp

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_test_single_dynamic_lstm_state_is_tuple(self):
        self.internal_test_single_dynamic_lstm(True)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_test_single_dynamic_lstm_state_is_not_tuple(self):
        self.internal_test_single_dynamic_lstm(False)

    def internal_test_single_dynamic_lstm(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_time_major(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                time_major=True,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_forget_bias(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                forget_bias=0.5)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                time_major=True,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_single_dynamic_lstm_seq_length_is_const(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        state_is_tuple = True
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=[4, 3, 4, 5, 2, 1])

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_single_dynamic_lstm_seq_length_is_not_const(self):
        for np_dtype in [np.int32, np.int64, np.float32]:
            units = 5
            batch_size = 6
            x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
            x_val = np.stack([x_val] * batch_size)
            y_val = np.array([4, 3, 4, 5, 2, 1], dtype=np_dtype)
            state_is_tuple = True
            def func(x, seq_length):
                initializer = init_ops.constant_initializer(0.5)

                # no scope
                cell = LSTMCell(
                    units,
                    initializer=initializer,
                    state_is_tuple=state_is_tuple)
                outputs, cell_state = dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=tf.identity(seq_length))

                return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

            feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
            input_names_with_port = ["input_1:0", "input_2:0"]
            output_names_with_port = ["output:0", "cell_state:0"]
            self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 6)
        state_is_tuple = True
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)  # by default zero initializer is used

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_ch_zero_state_initializer(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        state_is_tuple = True
        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)

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
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_consume_one_of_ch_tuple(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)
            state_is_tuple = True
            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), \
                   tf.identity(cell_state.c, name="cell_state_c")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state_c:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_single_dynamic_lstm_random_weights(self, state_is_tuple=True):
        hidden_size = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=42)

            # no scope
            cell = LSTMCell(
                hidden_size,
                initializer=initializer,
                state_is_tuple=state_is_tuple)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001)

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_single_dynamic_lstm_random_weights2(self, state_is_tuple=True):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            # no scope
            cell = LSTMCell(
                hidden_size,
                initializer=initializer,
                state_is_tuple=state_is_tuple)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.01)

    @check_opset_after_tf_version("1.15", 8, "might need Select")
    def test_multiple_dynamic_lstm_state_is_tuple(self):
        self.internal_test_multiple_dynamic_lstm_with_parameters(True)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_multiple_dynamic_lstm_state_is_not_tuple(self):
        self.internal_test_multiple_dynamic_lstm_with_parameters(False)

    def internal_test_multiple_dynamic_lstm_with_parameters(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            lstm_output_list = []
            lstm_cell_state_list = []
            # no scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)

            # given scope
            cell = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=[4, 4, 4, 4, 4, 4],
                    scope=scope)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)

            return tf.identity(lstm_output_list, name="output"), tf.identity(lstm_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           require_lstm_count=2)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    @skip_tf2()     # Still failing likely due to inconsistent random number initialization
    def test_dynamic_basiclstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            cell1 = BasicLSTMCell(
                units,
                state_is_tuple=True)
            outputs, cell_state = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-06)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_dynamic_lstm_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=True)

            outputs, _ = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-07)

    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    def test_dynamic_lstm_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            cell1 = LSTMCell(units, initializer=initializer, state_is_tuple=True)
            _, cell_state = dynamic_rnn(cell1, x, dtype=tf.float32)
            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_state_is_tuple(self):
        self.internal_test_dynamic_bilstm_with_parameters(True)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_state_is_not_tuple(self):
        self.internal_test_dynamic_bilstm_with_parameters(False)

    def internal_test_dynamic_bilstm_with_parameters(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bilstm, no scope
            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
            cell2 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_output_consumed_only(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bilstm, no scope
            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
            cell2 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, _ = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_state_consumed_only(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bilstm, no scope
            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
            cell2 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            _, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_outputs_partially_consumed(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            # bilstm, no scope
            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
            cell2 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            (output_fw, _), (_, state_bw) = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(output_fw, name="output"), tf.identity(state_bw, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_unknown_batch_size(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)

            cell1 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            cell2 = LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
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
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @skip_tf_versions("2.1", "Bug in TF 2.1")
    def test_dynamic_multi_bilstm_with_same_input_hidden_size(self):
        batch_size = 10
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer1 = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            initializer2 = tf.random_uniform_initializer(0.0, 1.0, seed=43)
            initializer3 = tf.random_uniform_initializer(0.0, 1.0, seed=44)
            initializer4 = tf.random_uniform_initializer(0.0, 1.0, seed=45)
            units = 5
            cell1 = LSTMCell(units, name="cell1", initializer=initializer1)
            cell2 = LSTMCell(units, name="cell2", initializer=initializer2)
            outputs_1, cell_state_1 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32,
                scope="bilstm_1"
            )

            units = 10
            cell3 = LSTMCell(units, name="cell3", initializer=initializer3)
            cell4 = LSTMCell(units, name="cell4", initializer=initializer4)
            outputs_2, cell_state_2 = bidirectional_dynamic_rnn(
                cell3,
                cell4,
                x,
                dtype=tf.float32,
                scope="bilstm_2"
            )

            return tf.identity(outputs_1, name="output_1"), \
                   tf.identity(cell_state_1, name="cell_state_1"), \
                   tf.identity(outputs_2, name="output_2"), \
                   tf.identity(cell_state_2, name="cell_state_2")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output_1:0", "cell_state_1:0", "output_2:0", "cell_state_2:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           require_lstm_count=2)

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @skip_tf_versions("2.1", "Bug in TF 2.1")
    def test_dynamic_multi_bilstm_with_same_input_seq_len(self):
        units = 5
        batch_size = 10
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        seq_len_val = np.array([3], dtype=np.int32)

        def func(x, y1, y2):
            initializer1 = tf.random_uniform_initializer(0.0, 1.0, seed=42)
            initializer2 = tf.random_uniform_initializer(0.0, 1.0, seed=43)
            seq_len1 = tf.tile(y1, [batch_size])
            cell1 = LSTMCell(units, initializer=initializer1)
            cell2 = LSTMCell(units, initializer=initializer2)
            outputs_1, cell_state_1 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                sequence_length=seq_len1,
                dtype=tf.float32,
                scope="bilstm_1"
            )
            initializer1 = tf.random_uniform_initializer(0.0, 1.0, seed=44)
            initializer2 = tf.random_uniform_initializer(0.0, 1.0, seed=45)
            seq_len2 = tf.tile(y2, [batch_size])
            cell1 = LSTMCell(units, initializer=initializer1)
            cell2 = LSTMCell(units, initializer=initializer2)
            outputs_2, cell_state_2 = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                sequence_length=seq_len2,
                dtype=tf.float32,
                scope="bilstm_2"
            )

            return tf.identity(outputs_1, name="output_1"), \
                   tf.identity(cell_state_1, name="cell_state_1"), \
                   tf.identity(outputs_2, name="output_2"), \
                   tf.identity(cell_state_2, name="cell_state_2")

        feed_dict = {"input_1:0": x_val, "input_2:0": seq_len_val, "input_3:0": seq_len_val}
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        output_names_with_port = ["output_1:0", "cell_state_1:0", "output_2:0", "cell_state_2:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           require_lstm_count=2)

    @check_tf_min_version("2.0")
    @skip_tf_versions("2.1", "Bug in TF 2.1")
    def test_keras_lstm(self):
        in_shape = [10, 3]
        x_val = np.random.uniform(size=[2, 10, 3]).astype(np.float32)

        model_in = tf.keras.layers.Input(tuple(in_shape), batch_size=2)
        x = tf.keras.layers.LSTM(
            units=5,
            return_sequences=True,
            return_state=True,
            kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
            recurrent_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
            bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43)
        )(model_in)
        model = tf.keras.models.Model(inputs=model_in, outputs=x)

        def func(x):
            y = model(x)
            # names for input and outputs for tests
            return tf.identity(y[0], name="output"), tf.identity(y[1], name="output1")
        self.run_test_case(func, {"input:0": x_val}, [], ["output:0", "output1:0"], rtol=1e-05, atol=1e-06)

    @check_tf_min_version("2.0")
    @skip_tf_versions("2.1", "Bug in TF 2.1")
    def test_keras_lstm_recurrent_activation_is_hard_sigmoid(self):
        in_shape = [10, 3]
        x_val = np.random.uniform(size=[2, 10, 3]).astype(np.float32)

        model_in = tf.keras.layers.Input(tuple(in_shape), batch_size=2)
        x = tf.keras.layers.LSTM(
            units=5,
            return_sequences=True,
            return_state=True,
            kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
            recurrent_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
            bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43),
            recurrent_activation="hard_sigmoid"
        )(model_in)
        model = tf.keras.models.Model(inputs=model_in, outputs=x)

        def func(x):
            y = model(x)
            return tf.identity(y[0], name="output"), tf.identity(y[1], name="output1")
        self.run_test_case(func, {"input:0": x_val}, [], ["output:0", "output1:0"], rtol=1e-05, atol=1e-06)

    @check_tf_min_version("2.0")
    def test_keras_bilstm_recurrent_activation_is_hard_sigmoid(self):
        in_shape = [10, 3]
        x_val = np.random.uniform(size=[2, 10, 3]).astype(np.float32)

        model_in = tf.keras.layers.Input(tuple(in_shape), batch_size=2)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=5,
                return_sequences=True,
                return_state=True,
                kernel_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=42),
                recurrent_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=44),
                bias_initializer=tf.random_uniform_initializer(0.0, 1.0, seed=43),
                recurrent_activation="hard_sigmoid",
            )
        )(model_in)
        model = tf.keras.models.Model(inputs=model_in, outputs=x)

        def func(x):
            y = model(x)
            return tf.identity(y[0], name="output"), tf.identity(y[1], name="output1")
        self.run_test_case(func, {"input:0": x_val}, [], ["output:0", "output1:0"], rtol=1e-05, atol=1e-06)

    @check_tf_min_version("2.0")
    @skip_tfjs("TFJS converts model incorrectly")
    def test_keras_lstm_sigmoid_dropout(self):
        in_shape = [16, 16]
        batch_size = 2
        x_val = np.random.uniform(size=[batch_size] + in_shape).astype(np.float32)

        model = tf.keras.models.Sequential()
        model_in = tf.keras.layers.Input(shape=tuple(in_shape), name="input")
        lstm = tf.keras.layers.LSTM(16, activation='sigmoid', dropout=0.1)
        model.add(model_in)
        model.add(lstm)

        def func(x):
            y = model(x)
            return tf.identity(y[0], name="output")
        self.run_test_case(func, {"input:0": x_val}, [], ["output:0"], rtol=1e-05, atol=1e-06)

if __name__ == '__main__':
    unittest_main()
