# SPDX-License-Identifier: Apache-2.0

import onnx
import pytest
import numpy as np

import keras2onnx
from keras2onnx import common as _cmn
from keras2onnx.proto import keras
from onnxconverter_common.onnx_ex import get_maximum_opset_supported

from distutils.version import StrictVersion

"""
Tests for ONNX Operator Builder
"""


def test_apply():
    oopb = _cmn.onnx_ops.OnnxOperatorBuilder(_cmn.OnnxObjectContainer(get_maximum_opset_supported()),
                                             _cmn.InterimContext('_curr'))
    value = oopb.apply_add((np.array([[1.0], [0.5]], dtype='float32'),
                            ('_i1', oopb.float, np.array([2.0], dtype='float32'))), 'add')
    assert value[0].startswith('add')


@pytest.mark.skipif(StrictVersion(onnx.__version__) < StrictVersion("1.2"),
                    reason="Not supported in ONNX version less than 1.2, since this test requires opset 7.")
def test_model_creation():
    N, C, H, W = 2, 3, 5, 5
    input1 = keras.layers.Input(shape=(H, W, C))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(H, W, C))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    maximum_layer = keras.layers.Maximum()([x1, x2])

    out = keras.layers.Dense(8)(maximum_layer)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    trial1 = np.random.rand(N, H, W, C).astype(np.float32, copy=False)
    trial2 = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

    predicted = model.predict([trial1, trial2])
    assert predicted is not None

    converted_model_7 = keras2onnx.convert_keras(model, target_opset=7)
    converted_model_5 = keras2onnx.convert_keras(model, target_opset=5)

    assert converted_model_7 is not None
    assert converted_model_5 is not None

    assert converted_model_7.opset_import[0].version > converted_model_5.opset_import[0].version
