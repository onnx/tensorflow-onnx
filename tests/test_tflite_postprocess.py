# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for TFLite_Detection_PostProcess op"""

import os
import struct
import numpy as np
import flatbuffers

from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from backend_test_base import Tf2OnnxBackendTestBase

from tf2onnx import utils
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import optimizer

from tf2onnx.tflite import Model, OperatorCode, SubGraph, Operator, Tensor, Buffer
from tf2onnx.tflite.BuiltinOperator import BuiltinOperator
from tf2onnx.tflite.TensorType import TensorType
from tf2onnx.tflite.CustomOptionsFormat import CustomOptionsFormat

# pylint: disable=missing-docstring

def endvector(builder, length):
    try:
        return builder.EndVector(length)
    except TypeError:
        # flatbuffers 2.0 changes the API
        return builder.EndVector()

class TFLiteDetectionPostProcessTests(Tf2OnnxBackendTestBase):

    @requires_tflite("TFLite_Detection_PostProcess")
    @check_opset_min_version(11, "Pad")
    def test_postprocess_model1(self):
        self._test_postprocess(num_classes=5, num_boxes=100, detections_per_class=2, max_detections=20)

    @requires_tflite("TFLite_Detection_PostProcess")
    @check_opset_min_version(11, "Pad")
    def test_postprocess_model2(self):
        self._test_postprocess(num_classes=5, num_boxes=100, detections_per_class=7, max_detections=20)

    @requires_tflite("TFLite_Detection_PostProcess")
    @check_opset_min_version(11, "Pad")
    def test_postprocess_model3(self):
        self._test_postprocess(num_classes=5, num_boxes=3, detections_per_class=7, max_detections=20)

    @requires_tflite("TFLite_Detection_PostProcess")
    @check_opset_min_version(11, "Pad")
    def test_postprocess_model4(self):
        self._test_postprocess(num_classes=5, num_boxes=99, detections_per_class=2, max_detections=20, extra_class=True)

    @requires_tflite("TFLite_Detection_PostProcess")
    @check_opset_min_version(11, "Pad")
    def test_postprocess_model5(self):
        self._test_postprocess(num_classes=1, num_boxes=100, detections_per_class=0,
                               max_detections=50, use_regular_nms=False)

    def _test_postprocess(self, num_classes, num_boxes, detections_per_class,
                          max_detections, extra_class=False, use_regular_nms=True):
        model = self.make_postprocess_model(num_classes=num_classes, detections_per_class=detections_per_class,
                                            max_detections=max_detections, x_scale=11.0, w_scale=6.0,
                                            use_regular_nms=use_regular_nms)

        np.random.seed(42)
        box_encodings_val = np.random.random_sample([1, num_boxes, 4]).astype(np.float32)
        if extra_class:
            num_classes += 1
        class_predictions_val = np.random.random_sample([1, num_boxes, num_classes]).astype(np.float32)
        anchors_val = np.random.random_sample([num_boxes, 4]).astype(np.float32)

        feed_dict = {
            "box_encodings": box_encodings_val,
            "class_predictions": class_predictions_val,
            "anchors": anchors_val
        }

        self.run_tflite_test(model, feed_dict)

    def make_postprocess_model(self, max_detections=10, detections_per_class=100, max_classes_per_detection=1,
                               use_regular_nms=True, nms_score_threshold=0.3, nms_iou_threshold=0.6, num_classes=90,
                               x_scale=10.0, y_scale=10.0, w_scale=5.0, h_scale=5.0):
        """Returns the bytes of a tflite model containing a single TFLite_Detection_PostProcess op"""

        builder = flatbuffers.Builder(1024)

        # op_code
        custom_code = builder.CreateString("TFLite_Detection_PostProcess")
        OperatorCode.OperatorCodeStart(builder)
        OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(builder, BuiltinOperator.CUSTOM)
        OperatorCode.OperatorCodeAddCustomCode(builder, custom_code)
        OperatorCode.OperatorCodeAddBuiltinCode(builder, BuiltinOperator.CUSTOM)
        op_code = OperatorCode.OperatorCodeEnd(builder)

        # op_codes
        Model.ModelStartOperatorCodesVector(builder, 1)
        builder.PrependUOffsetTRelative(op_code)
        op_codes = endvector(builder, 1)

        # Make tensors
        # [names, shape, type tensors]
        ts = []
        inputs_info = [('box_encodings', [-1, -1, 4]), ('class_predictions', [-1, -1, -1]), ('anchors', [-1, 4])]
        outputs_info = [
            ('detection_boxes', [-1, -1, 4]),
            ('detection_classes', [-1, -1]),
            ('detection_scores', [-1, -1]),
            ('num_detections', [-1])
        ]
        for name_info, shape_info in inputs_info + outputs_info:

            name = builder.CreateString(name_info)
            shape = builder.CreateNumpyVector(np.maximum(np.array(shape_info, np.int32), 1))
            shape_signature = builder.CreateNumpyVector(np.array(shape_info, np.int32))

            Tensor.TensorStart(builder)
            Tensor.TensorAddShape(builder, shape)
            Tensor.TensorAddType(builder, TensorType.FLOAT32)
            Tensor.TensorAddName(builder, name)
            Tensor.TensorAddShapeSignature(builder, shape_signature)
            ts.append(Tensor.TensorEnd(builder))

        SubGraph.SubGraphStartTensorsVector(builder, len(ts))
        for tensor in reversed(ts):
            builder.PrependUOffsetTRelative(tensor)
        tensors = endvector(builder, len(ts))

        # inputs
        SubGraph.SubGraphStartInputsVector(builder, 3)
        for inp in reversed([0, 1, 2]):
            builder.PrependInt32(inp)
        inputs = endvector(builder, 3)

        # outputs
        SubGraph.SubGraphStartOutputsVector(builder, 4)
        for out in reversed([3, 4, 5, 6]):
            builder.PrependInt32(out)
        outputs = endvector(builder, 4)

        flexbuffer = \
            b'y_scale\x00nms_score_threshold\x00max_detections\x00x_scale\x00w_scale\x00nms_iou_threshold' \
            b'\x00use_regular_nms\x00h_scale\x00max_classes_per_detection\x00num_classes\x00detections_per_class' \
            b'\x00\x0b\x16E>\x88j\x9e([v\x7f\xab\x0b\x00\x00\x00\x01\x00\x00\x00\x0b\x00\x00\x00*attr4**attr7*' \
            b'*attr10**attr9**attr1**attr2**attr3**attr11*\x00\x00\x00*attr8**attr5**attr6*\x06\x0e\x06\x06\x0e' \
            b'\x0e\x06j\x0e\x0e\x0e7&\x01'
        flexbuffer = flexbuffer.replace(b'*attr1*', struct.pack('<f', nms_iou_threshold))
        flexbuffer = flexbuffer.replace(b'*attr2*', struct.pack('<f', nms_score_threshold))
        flexbuffer = flexbuffer.replace(b'*attr3*', struct.pack('<i', num_classes))
        flexbuffer = flexbuffer.replace(b'*attr4*', struct.pack('<i', detections_per_class))
        flexbuffer = flexbuffer.replace(b'*attr5*', struct.pack('<f', x_scale))
        flexbuffer = flexbuffer.replace(b'*attr6*', struct.pack('<f', y_scale))
        flexbuffer = flexbuffer.replace(b'*attr7*', struct.pack('<f', h_scale))
        flexbuffer = flexbuffer.replace(b'*attr8*', struct.pack('<f', w_scale))
        flexbuffer = flexbuffer.replace(b'*attr9*', struct.pack('<i', max_detections))
        flexbuffer = flexbuffer.replace(b'*attr10*', struct.pack('<i', max_classes_per_detection))
        flexbuffer = flexbuffer.replace(b'*attr11*', struct.pack('<b', use_regular_nms))

        custom_options = builder.CreateNumpyVector(np.array(bytearray(flexbuffer)))

        # operator
        Operator.OperatorStart(builder)
        Operator.OperatorAddOpcodeIndex(builder, 0)
        Operator.OperatorAddInputs(builder, inputs)
        Operator.OperatorAddOutputs(builder, outputs)
        Operator.OperatorAddCustomOptions(builder, custom_options)
        Operator.OperatorAddCustomOptionsFormat(builder, CustomOptionsFormat.FLEXBUFFERS)
        operator = Operator.OperatorEnd(builder)

        # operators
        SubGraph.SubGraphStartOperatorsVector(builder, 1)
        builder.PrependUOffsetTRelative(operator)
        operators = endvector(builder, 1)

        # subgraph
        graph_name = builder.CreateString("TFLite graph")
        SubGraph.SubGraphStart(builder)
        SubGraph.SubGraphAddName(builder, graph_name)
        SubGraph.SubGraphAddTensors(builder, tensors)
        SubGraph.SubGraphAddInputs(builder, inputs)
        SubGraph.SubGraphAddOutputs(builder, outputs)
        SubGraph.SubGraphAddOperators(builder, operators)
        subgraph = SubGraph.SubGraphEnd(builder)

        # subgraphs
        Model.ModelStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph)
        subgraphs = endvector(builder, 1)

        description = builder.CreateString("Model for tflite testing")

        Buffer.BufferStartDataVector(builder, 0)
        data = endvector(builder, 0)

        Buffer.BufferStart(builder)
        Buffer.BufferAddData(builder, data)
        buffer = Buffer.BufferEnd(builder)

        Model.ModelStartBuffersVector(builder, 1)
        builder.PrependUOffsetTRelative(buffer)
        buffers = endvector(builder, 1)

        # model
        Model.ModelStart(builder)
        Model.ModelAddVersion(builder, 3)
        Model.ModelAddOperatorCodes(builder, op_codes)
        Model.ModelAddSubgraphs(builder, subgraphs)
        Model.ModelAddDescription(builder, description)
        Model.ModelAddBuffers(builder, buffers)
        model = Model.ModelEnd(builder)

        builder.Finish(model, b"TFL3")
        return builder.Output()

    def run_tflite_test(self, tflite_model, feed_dict, rtol=1e-07, atol=1e-5):
        tflite_path = os.path.join(self.test_data_directory, self._testMethodName + ".tflite")
        dir_name = os.path.dirname(tflite_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        tf_lite_output_data, output_names = self.run_tflite(tflite_path, feed_dict)

        g = process_tf_graph(None, opset=self.config.opset,
                             input_names=list(feed_dict.keys()),
                             output_names=output_names,
                             target=self.config.target,
                             tflite_path=tflite_path)
        g = optimizer.optimize_graph(g)
        onnx_from_tfl_output = self.run_backend(g, output_names, feed_dict, postfix="_from_tflite")
        self.assert_results_equal(tf_lite_output_data, onnx_from_tfl_output, rtol, atol)
