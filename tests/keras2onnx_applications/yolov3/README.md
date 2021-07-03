<!--- SPDX-License-Identifier: Apache-2.0 -->

# Introduction
The original keras model was coming from: <https://github.com/qqwweee/keras-yolo3>, clone the project and follow the 'Quick Start' to get the pre-trained model.

We have converted yolov3 model successfully and uploaded to the model zoo <https://github.com/onnx/models/tree/master/yolov3>

The model supports `batch_size = 1`.

# Convert
```
export PYTHONPATH=$(the keras-yolo3 path)
# run object detection, convert the model to onnx first if the onnx model does not exist
python yolov3.py <image url>
```
The unit test is added in our nightly build, see [here](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_yolov3.py)
