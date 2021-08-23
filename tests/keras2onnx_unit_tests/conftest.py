# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import numpy as np
import tensorflow as tf

from mock_keras2onnx.proto import keras, is_tf_keras
from test_utils import run_onnx_runtime
from mock_keras2onnx.proto.tfcompat import is_tf2

K = keras.backend


@pytest.fixture(scope='function')
def runner():
    np.random.seed(42)
    if is_tf2:
        tf.random.set_seed(42)
    else:
        tf.random.set_random_seed(42)
    model_files = []

    def runner_func(*args, **kwargs):
        return run_onnx_runtime(*args, model_files, **kwargs)

    # Ensure Keras layer naming is reset for each function
    K.reset_uids()
    # Reset the TensorFlow session to avoid resource leaking between tests
    K.clear_session()

    # Provide wrapped run_onnx_runtime function
    yield runner_func
    # Remove model files
    for fl in model_files:
        os.remove(fl)
