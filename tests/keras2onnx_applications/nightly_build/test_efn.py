# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from os.path import dirname, abspath
from mock_keras2onnx.proto import keras, is_tensorflow_older_than

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_image

img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


@unittest.skipIf(is_tensorflow_older_than('2.1.0'), "efficientnet needs tensorflow >= 2.1.0")
class TestEfn(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skip("TODO: model discrepancy")
    def test_custom(self):
        from efficientnet import tfkeras as efn
        keras.backend.set_learning_phase(0)
        base_model = efn.EfficientNetB0(input_shape=(600, 600, 3), weights=None)
        backbone = keras.Model(base_model.input, base_model.get_layer("top_activation").output)
        res = run_image(backbone, self.model_files, img_path, target_size=(600, 600),
                        rtol=1e-2, atol=1e-1)
        self.assertTrue(*res)

    def test_efn(self):
        from efficientnet import tfkeras as efn
        keras.backend.set_learning_phase(0)
        model = efn.EfficientNetB0(weights=None)
        res = run_image(model, self.model_files, img_path, target_size=(224, 224), rtol=1e-2)
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
