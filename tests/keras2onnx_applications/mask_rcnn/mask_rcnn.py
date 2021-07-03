# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import skimage
import onnx
import mock_keras2onnx

from mrcnn.config import Config
from mrcnn.model import BatchNorm, DetectionLayer
from mrcnn import model as modellib
from mrcnn import visualize

from mock_keras2onnx import set_converter
from mock_keras2onnx.ke2onnx.batch_norm import convert_keras_batch_normalization
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import convert_tf_crop_and_resize


ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


def convert_BatchNorm(scope, operator, container):
    convert_keras_batch_normalization(scope, operator, container)


def norm_boxes_graph(scope, operator, container, oopb, image_meta):
    image_shapes = oopb.add_node('Slice',
                         [image_meta,
                          ('_start', oopb.int64, np.array([4], dtype='int64')),
                          ('_end', oopb.int64, np.array([7], dtype='int64')),
                          ('_axes', oopb.int64, np.array([1], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_image_shapes')
    image_shape = oopb.add_node('Slice',
                                 [image_shapes,
                                  ('_start', oopb.int64, np.array([0], dtype='int64')),
                                  ('_end', oopb.int64, np.array([1], dtype='int64')),
                                  ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                  ],
                                 operator.inputs[0].full_name + '_image_shape')
    image_shape_squeeze = oopb.apply_squeeze(image_shape, name=operator.full_name + '_image_shape_squeeze', axes=[0])[0]

    window = oopb.add_node('Slice',
                            [image_meta,
                             ('_start', oopb.int64, np.array([7], dtype='int64')),
                             ('_end', oopb.int64, np.array([11], dtype='int64')),
                             ('_axes', oopb.int64, np.array([1], dtype='int64'))
                             ],
                            operator.inputs[0].full_name + '_window')
    h_norm = oopb.add_node('Slice',
                         [image_shape_squeeze,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_h_norm')
    w_norm = oopb.add_node('Slice',
                           [image_shape_squeeze,
                            ('_start', oopb.int64, np.array([1], dtype='int64')),
                            ('_end', oopb.int64, np.array([2], dtype='int64')),
                            ('_axes', oopb.int64, np.array([0], dtype='int64'))
                            ],
                           operator.inputs[0].full_name + '_w_norm')
    h_norm_float = scope.get_unique_variable_name('h_norm_float')
    attrs = {'to': 1}
    container.add_node('Cast', h_norm, h_norm_float, op_version=operator.target_opset,
                       **attrs)
    w_norm_float = scope.get_unique_variable_name('w_norm_float')
    attrs = {'to': 1}
    container.add_node('Cast', w_norm, w_norm_float, op_version=operator.target_opset,
                       **attrs)
    hw_concat = scope.get_unique_variable_name(operator.inputs[0].full_name + '_hw_concat')
    attrs = {'axis': -1}
    container.add_node("Concat",
                       [h_norm_float, w_norm_float, h_norm_float, w_norm_float],
                       hw_concat,
                       op_version=operator.target_opset,
                       name=operator.inputs[0].full_name + '_hw_concat', **attrs)
    scale = oopb.add_node('Sub',
                          [hw_concat,
                           ('_sub', oopb.float, np.array([1.0], dtype='float32'))
                           ],
                          operator.inputs[0].full_name + '_scale')
    boxes_shift = oopb.add_node('Sub',
                          [window,
                           ('_sub', oopb.float, np.array([0.0, 0.0, 1.0, 1.0], dtype='float32'))
                           ],
                          operator.inputs[0].full_name + '_boxes_shift')
    divide = oopb.add_node('Div',
                            [boxes_shift, scale],
                            operator.inputs[0].full_name + '_divide')
    # output shape: [batch, 4]
    return divide


def convert_DetectionLayer(scope, operator, container):
    # type: (mock_keras2onnx.common.InterimContext, mock_keras2onnx.common.Operator, mock_keras2onnx.common.OnnxObjectContainer) -> None
    pass


set_converter(DetectionLayer, convert_DetectionLayer)
set_converter(BatchNorm, convert_BatchNorm)


# Run detection
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def generate_image(images, molded_images, windows, results):
    results_final = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            model.unmold_detections(results[0][i], results[3][i], # detections[i], mrcnn_mask[i]
                                   image.shape, molded_images[i].shape,
                                   windows[i])
        results_final.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
        r = results_final[i]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    return results_final


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need an image file for object detection.")
        exit(-1)

    model_file_name = './mrcnn.onnx'
    if not os.path.exists(model_file_name):
        # use opset 11 or later
        set_converter('CropAndResize', convert_tf_crop_and_resize)
        oml = mock_keras2onnx.convert_keras(model.keras_model, target_opset=11)
        onnx.save_model(oml, model_file_name)

    # run with ONNXRuntime
    import onnxruntime
    filename = sys.argv[1]
    image = skimage.io.imread(filename)
    images = [image]

    sess = onnxruntime.InferenceSession(model_file_name)

    # preprocessing
    molded_images, image_metas, windows = model.mold_inputs(images)
    anchors = model.get_anchors(molded_images[0].shape)
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

    results = \
        sess.run(None, {"input_image": molded_images.astype(np.float32),
                        "input_anchors": anchors,
                        "input_image_meta": image_metas.astype(np.float32)})

    # postprocessing
    results_final = generate_image(images, molded_images, windows, results)
