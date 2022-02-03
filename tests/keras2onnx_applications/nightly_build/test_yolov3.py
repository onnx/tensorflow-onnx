# SPDX-License-Identifier: Apache-2.0

import os
from os.path import dirname, abspath
import numpy as np
import unittest
import sys
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../yolov3'))

from keras.models import load_model
import onnx
import urllib.request
from yolov3 import YOLO, convert_model

from distutils.version import StrictVersion
import mock_keras2onnx
from test_utils import is_bloburl_access


YOLOV3_WEIGHTS_PATH = r'https://lotus.blob.core.windows.net/converter-models/yolov3.h5'
model_file_name = 'yolov3.h5'
YOLOV3_TINY_WEIGHTS_PATH = r'https://lotus.blob.core.windows.net/converter-models/yolov3-tiny.h5'
tiny_model_file_name = 'yolov3-tiny.h5'

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class TestYoloV3(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def post_compute(self, all_boxes, all_scores, indices):
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices[0]:
            out_classes.append(idx_[1])
            out_scores.append(all_scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(all_boxes[idx_1])
        return [out_boxes, out_scores, out_classes]

    @unittest.skipIf(StrictVersion(onnx.__version__.split('-')[0]) < StrictVersion("1.5.0"),
                     "NonMaxSuppression op is not supported for onnx < 1.5.0.")
    @unittest.skipIf(not is_bloburl_access(YOLOV3_WEIGHTS_PATH) or not is_bloburl_access(YOLOV3_TINY_WEIGHTS_PATH),
                     "Model blob url can't access.")
    def test_yolov3(self):
        img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
        yolo3_yolo3_dir = os.path.join(os.path.dirname(__file__), '../../../keras-yolo3/yolo3')
        try:
            import onnxruntime
        except ImportError:
            return True

        from PIL import Image

        for is_tiny_yolo in [True, False]:
            if is_tiny_yolo:
                if not os.path.exists(tiny_model_file_name):
                    urllib.request.urlretrieve(YOLOV3_TINY_WEIGHTS_PATH, tiny_model_file_name)
                yolo_weights = load_model(tiny_model_file_name)
                model_path = tiny_model_file_name  # model path or trained weights path
                anchors_path = 'model_data/tiny_yolo_anchors.txt'
                case_name = 'yolov3-tiny'
            else:
                if not os.path.exists(model_file_name):
                    urllib.request.urlretrieve(YOLOV3_WEIGHTS_PATH, model_file_name)
                yolo_weights = load_model(model_file_name)
                model_path = model_file_name  # model path or trained weights path
                anchors_path = 'model_data/yolo_anchors.txt'
                case_name = 'yolov3'

            my_yolo = YOLO(model_path, anchors_path, yolo3_yolo3_dir)
            my_yolo.load_model(yolo_weights)
            onnx_model = convert_model(my_yolo, is_tiny_yolo)

            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
            onnx.save_model(onnx_model, temp_model_file)

            sess = onnxruntime.InferenceSession(temp_model_file)

            image = Image.open(img_path)
            image_data = my_yolo.prepare_keras_data(image)

            all_boxes_k, all_scores_k, indices_k = my_yolo.final_model.predict([image_data, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2)])

            image_data_onnx = np.transpose(image_data, [0, 3, 1, 2])

            feed_f = dict(zip(['input_1', 'image_shape'],
                              (image_data_onnx, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2))))
            all_boxes, all_scores, indices = sess.run(None, input_feed=feed_f)

            expected = self.post_compute(all_boxes_k, all_scores_k, indices_k)
            actual = self.post_compute(all_boxes, all_scores, indices)

            res = all(np.allclose(expected[n_], actual[n_]) for n_ in range(3))
            self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
