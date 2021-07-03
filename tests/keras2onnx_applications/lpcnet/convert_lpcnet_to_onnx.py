# SPDX-License-Identifier: Apache-2.0

import lpcnet
import sys

model, enc, dec = lpcnet.new_lpcnet_model(use_gpu=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model_file = sys.argv[1]
model.load_weights(model_file)

import mock_keras2onnx
oxml_enc = mock_keras2onnx.convert_keras(enc, 'lpcnet_enc')
oxml_dec = mock_keras2onnx.convert_keras(dec, 'lpcnet_dec')

import onnx
onnx.save(oxml_enc, "lpcnet_enc.onnx")
onnx.save(oxml_dec, "lpcnet_dec.onnx")
