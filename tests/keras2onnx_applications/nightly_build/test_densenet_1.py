# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import keras_segmentation
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_image

img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

from mock_keras2onnx.proto import is_keras_older_than

class TestDenseNet_1(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(is_keras_older_than("2.2.3"),
                     "Cannot import normalize_data_format from keras.backend")
    def test_densenet(self):
        # From https://github.com/titu1994/DenseNet/blob/master/densenet.py
        sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../model_source/densenet_1/'))
        import densenet_1
        image_dim = (224, 224, 3)
        model = densenet_1.DenseNetImageNet121(input_shape=image_dim)
        res = run_image(model, self.model_files, img_path, target_size=(224, 224))
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
