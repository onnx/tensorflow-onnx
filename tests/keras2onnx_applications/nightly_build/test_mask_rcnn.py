# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
from mock_keras2onnx import set_converter
from mock_keras2onnx.proto import keras
import onnx
import numpy as np
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime, print_mismatches, convert_tf_crop_and_resize

import urllib.request
MASKRCNN_WEIGHTS_PATH = r'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5'
model_file_name = 'mask_rcnn_coco.h5'
if not os.path.exists(model_file_name):
    urllib.request.urlretrieve(MASKRCNN_WEIGHTS_PATH, model_file_name)

keras.backend.clear_session()
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../mask_rcnn/'))
from mask_rcnn import model
from distutils.version import StrictVersion

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')

# mask rcnn code From https://github.com/matterport/Mask_RCNN
class TestMaskRCNN(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(StrictVersion(onnx.__version__.split('-')[0]) < StrictVersion("1.6.0"),
                     "Mask-rcnn conversion needs contrib op for onnx < 1.6.0.")
    def test_mask_rcnn(self):
        set_converter('CropAndResize', convert_tf_crop_and_resize)
        onnx_model = mock_keras2onnx.convert_keras(model.keras_model)

        import skimage
        img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
        image = skimage.io.imread(img_path)
        images = [image]
        case_name = 'mask_rcnn'

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)
        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        # preprocessing
        molded_images, image_metas, windows = model.mold_inputs(images)
        anchors = model.get_anchors(molded_images[0].shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        expected = model.keras_model.predict(
            [molded_images.astype(np.float32), image_metas.astype(np.float32), anchors])

        actual = \
            sess.run(None, {"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})

        rtol = 1.e-3
        atol = 1.e-6
        compare_idx = [0, 3]
        res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in compare_idx)
        if res and temp_model_file not in self.model_files:  # still keep the failed case files for the diagnosis.
            self.model_files.append(temp_model_file)
        if not res:
            for n_ in compare_idx:
                expected_list = expected[n_].flatten()
                actual_list = actual[n_].flatten()
                print_mismatches(case_name, n_, expected_list, actual_list, atol, rtol)

        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
