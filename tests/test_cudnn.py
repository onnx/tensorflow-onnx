# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for cudnn."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops
from backend_test_base import Tf2OnnxBackendTestBase
from common import check_tf_max_version, skip_tf_cpu, check_opset_min_version, unittest_main


class CudnnTests(Tf2OnnxBackendTestBase):
    """ test cudnn cases """
    @check_tf_max_version("1.15.0", "not supported in tf-2.0")
    @skip_tf_cpu("only tf_gpu can run CudnnGPU")
    @check_opset_min_version(10, "CudnnGRU")
    def test_cudnngru(self):
        """ test contrib cudnn gru """
        seq_length = 3
        batch_size = 5
        input_size = 2
        num_layers = 2
        num_units = 2
        num_dirs = 2
        x_val = np.random.randint(0, 100, [seq_length, batch_size, input_size]).astype(np.float32)
        h_val = np.random.randint(0, 100, [num_layers * num_dirs, batch_size, num_units]).astype(np.float32).reshape(
            [num_layers * num_dirs, batch_size, num_units])

        def func(x, h):
            initializer = init_ops.constant_initializer(0.5)
            cudnngru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units, 'linear_input', 'bidirectional',
                                                     kernel_initializer=initializer, bias_initializer=initializer)
            cudnngru.build([seq_length, batch_size, input_size])
            outputs = cudnngru.call(x, tuple([h]))
            _ = tf.identity(outputs[0], name='output')

        feed_dict = {"input_1:0": x_val, "input_2:0": h_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-05, atol=1e-04)


if __name__ == '__main__':
    unittest_main()
