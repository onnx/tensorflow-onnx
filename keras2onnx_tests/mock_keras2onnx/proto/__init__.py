# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import tensorflow
from distutils.version import StrictVersion

# Rather than using ONNX protobuf definition throughout our codebase, we import ONNX protobuf definition here so that
# we can conduct quick fixes by overwriting ONNX functions without changing any lines elsewhere.
from onnx import onnx_pb as onnx_proto
from onnx import helper
from onnx import save_model as save_model


def _check_onnx_version():
    import pkg_resources
    min_required_version = pkg_resources.parse_version('1.0.1')
    current_version = pkg_resources.get_distribution('onnx').parsed_version
    assert current_version >= min_required_version, 'Keras2ONNX requires ONNX version 1.0.1 or a newer one'


_check_onnx_version()


def is_tensorflow_older_than(version_str):
    return StrictVersion(tensorflow.__version__.split('-')[0]) < StrictVersion(version_str)


def is_tensorflow_later_than(version_str):
    return StrictVersion(tensorflow.__version__.split('-')[0]) > StrictVersion(version_str)


is_tf_keras = False
str_tk_keras = os.environ.get('TF_KERAS', None)
if str_tk_keras is None:
    # With tensorflow 2.x, be default we loaded tf.keras as the framework, instead of Keras
    is_tf_keras = not is_tensorflow_older_than('2.0.0')
else:
    is_tf_keras = str_tk_keras != '0'

if is_tf_keras:
    from tensorflow.python import keras
else:
    try:
        import keras

        if keras.Model == tensorflow.keras.Model:  # since keras 2.4, keras and tf.keras is unified.
            is_tf_keras = True
    except ImportError:
        is_tf_keras = True
        from tensorflow.python import keras


def is_keras_older_than(version_str):
    return StrictVersion(keras.__version__.split('-')[0]) < StrictVersion(version_str)


def is_keras_later_than(version_str):
    return StrictVersion(keras.__version__.split('-')[0]) > StrictVersion(version_str)
