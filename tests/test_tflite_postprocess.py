# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for TFLite_Detection_PostProcess op"""

import os
import tensorflow as tf
import numpy as np

from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from backend_test_base import Tf2OnnxBackendTestBase
from tf2onnx.tf_loader import from_function, tf_session
from tf2onnx.tflite_utils import read_tflite_model, parse_tflite_graph

from tf2onnx import utils
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import optimizer

# pylint: disable=missing-docstring


_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFINPUT3 = "input3"
_INPUT3 = "input3:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"
_TFOUTPUT2 = "output2"
_OUTPUT2 = "output2:0"


class TFLiteDetectionPostProcessTests(Tf2OnnxBackendTestBase):

    @check_tf_min_version("2.0")
    def test_relu6_tflite(self):
        tflite_model = b'\x1c\x00\x00\x00TFL3\x14\x00 \x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x00\x00\x18\x00\x1c\x00\x14\x00\x00\x00\x03\x00\x00\x00\x18\x00\x00\x00\x1c\x00\x00\x00\x90\x00\x00\x00\x1c\x00\x00\x000\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x18\x01\x00\x00\x01\x00\x00\x00\x98\x00\x00\x00\x04\x00\x00\x00\x90\x01\x00\x00\x8c\x01\x00\x00\x88\x01\x00\x00D\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x0c\x00\x04\x00\x08\x00\x08\x00\x00\x00\x08\x00\x00\x00\x03\x00\x00\x00\x13\x00\x00\x00min_runtime_version\x00\x00\x00\x06\x00\x08\x00\x04\x00\x06\x00\x00\x00\x04\x00\x00\x00\x10\x00\x00\x001.5.0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00MLIR Converted.\x00\x00\x00\x0e\x00\x18\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x0e\x00\x00\x00\x14\x00\x00\x00\x1c\x00\x00\x00 \x00\x00\x00$\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\xb0\x00\x00\x00h\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00main\x00\x00\n\x00\x0c\x00\x00\x00\x04\x00\x08\x00\n\x00\x00\x00\x10\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x10\x00\x0b\x00\x00\x00\x0c\x00\x04\x00\x0c\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x15\x01\x00\x00\x00\xca\xff\xff\xff\x10\x00\x00\x00\x02\x00\x00\x00\x14\x00\x00\x00 \x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x06\x00\x00\x00output\x00\x00\x04\x00\x06\x00\x04\x00\x00\x00\x00\x00\x0e\x00\x14\x00\x04\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x0e\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x1c\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x05\x00\x00\x00input\x00\x00\x00\xfc\xff\xff\xff\x04\x00\x04\x00\x04\x00\x00\x00'
        x_val = np.array([0.5, 1.0, -0.5, -1.0, 6, 7], dtype=np.float32).reshape((2, 3))
        feed_dict = {_INPUT: x_val}
        self.run_tflite_test(tflite_model, [_OUTPUT], feed_dict)


    def test_postprocess_model1(self):
        box_num = 100
        classes_num = 5
        batch_dim = 1
        model = self.make_postprocess_model(num_classes=classes_num, detections_per_class=1, max_detections=20)

        np.random.seed(42)
        box_encodings_val = np.random.random_sample([batch_dim, box_num, 4]).astype(np.float32)
        class_predictions_val = np.random.random_sample([batch_dim, box_num, classes_num]).astype(np.float32)
        anchors_val = np.random.random_sample([box_num, 4]).astype(np.float32)

        feed_dict = {
            "box_encodings": box_encodings_val,
            "class_predictions": class_predictions_val,
            "anchors": anchors_val
        }

        self.run_tflite_test(model, None, feed_dict)

    
    def make_postprocess_model(self, max_detections=10, detections_per_class=100, max_classes_per_detection=1, 
                               use_regular_nms=True, nms_score_threshold=0.3, nms_iou_threshold=0.6, num_classes=90,
                               x_scale=10.0, y_scale=10.0, w_scale=5.0, h_scale=5.0):
        import flatbuffers
        from tf2onnx.tflite import Model, OperatorCode, SubGraph, Operator, Tensor, Buffer
        from tf2onnx.tflite.BuiltinOperator import BuiltinOperator
        from tf2onnx.tflite.TensorType import TensorType
        from tf2onnx.tflite.CustomOptionsFormat import CustomOptionsFormat
        import struct
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
        op_codes = builder.EndVector(1)

        # Make tensors
        # [names, shape, type tensors]
        ts = []
        inputs_info = [('box_encodings', [-1, -1, 4]), ('class_predictions', [-1, -1, -1]), ('anchors', [-1, 4])]
        outputs_info = [('detection_boxes', [-1, -1, 4]), ('detection_classes', [-1, -1]), ('detection_scores', [-1, -1]), ('num_detections', [-1])]
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
        tensors = builder.EndVector(len(ts))

        # inputs
        SubGraph.SubGraphStartInputsVector(builder, 3)
        for inp in reversed([0, 1, 2]):
            builder.PrependInt32(inp)
        inputs = builder.EndVector(3)

        # outputs
        SubGraph.SubGraphStartOutputsVector(builder, 4)
        for out in reversed([3, 4, 5, 6]):
            builder.PrependInt32(out)
        outputs = builder.EndVector(4)

        flexbuffer = b'y_scale\x00nms_score_threshold\x00max_detections\x00x_scale\x00w_scale\x00nms_iou_threshold\x00use_regular_nms\x00h_scale\x00max_classes_per_detection\x00num_classes\x00detections_per_class\x00\x0b\x16E>\x88j\x9e([v\x7f\xab\x0b\x00\x00\x00\x01\x00\x00\x00\x0b\x00\x00\x00d\x00\x00\x00\x00\x00\xa0!\x02\x00\x00\x00\n\x00\x00\x00\x9a\x99\x19?\x9a\x99\x99>Z\x00\x00\x00*\x00\x00\x00\x00\x00\xa0@\x00\x00 A\x00\x00 B\x06\x0e\x06\x06\x0e\x0e\x06j\x0e\x0e\x0e7&\x01'
        flexbuffer = flexbuffer.replace(b'\x9a\x99\x19?', struct.pack('<f', nms_iou_threshold))
        flexbuffer = flexbuffer.replace(b'\x9a\x99\x99>', struct.pack('<f', nms_score_threshold))
        flexbuffer = flexbuffer.replace(b'Z\x00\x00\x00', struct.pack('<i', num_classes))
        flexbuffer = flexbuffer.replace(b'd\x00\x00\x00', struct.pack('<i', detections_per_class))
        flexbuffer = flexbuffer.replace(b'\x00\x00 A', struct.pack('<f', x_scale))
        flexbuffer = flexbuffer.replace(b'\x00\x00 B', struct.pack('<f', y_scale))
        flexbuffer = flexbuffer.replace(b'\x00\x00\xa0!', struct.pack('<f', h_scale))
        flexbuffer = flexbuffer.replace(b'\x00\x00\xa0@', struct.pack('<f', w_scale))
        flexbuffer = flexbuffer.replace(b'\n\x00\x00\x00', struct.pack('<i', max_detections))
        flexbuffer = flexbuffer.replace(b'\x02\x00\x00\x00', struct.pack('<i', max_classes_per_detection))
        flexbuffer = flexbuffer.replace(b'*', struct.pack('<b', use_regular_nms))


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
        operators = builder.EndVector(1)

        # subgraph
        SubGraph.SubGraphStart(builder)
        SubGraph.SubGraphAddTensors(builder, tensors)
        SubGraph.SubGraphAddInputs(builder, inputs)
        SubGraph.SubGraphAddOutputs(builder, outputs)
        SubGraph.SubGraphAddOperators(builder, operators)
        subgraph = SubGraph.SubGraphEnd(builder)

        # subgraphs
        Model.ModelStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph)
        subgraphs = builder.EndVector(1)

        description = builder.CreateString("Model for tflite testing")

        Buffer.BufferStartDataVector(builder, 0)
        data = builder.EndVector(0)

        Buffer.BufferStart(builder)
        Buffer.BufferAddData(builder, data)
        buffer = Buffer.BufferEnd(builder)

        Model.ModelStartBuffersVector(builder, 1)
        builder.PrependUOffsetTRelative(buffer)
        buffers = builder.EndVector(1)

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


        

    def run_tflite_test(self, tflite_model, output_names_with_port, feed_dict, rtol=1e-07, atol=1e-5):
        tflite_path = os.path.join(self.test_data_directory, self._testMethodName + ".tflite")
        dir_name = os.path.dirname(tflite_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        onnx_feed_dict = feed_dict
        interpreter = tf.lite.Interpreter(tflite_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_name_to_index = {n['name'].split(':')[0]: n['index'] for n in input_details}
        ouput_name_to_index = {n['name'].split(':')[0]: n['index'] for n in output_details}
        feed_dict_without_port = {k.split(':')[0]: v for k, v in feed_dict.items()}

        for k, v in feed_dict_without_port.items():
            interpreter.resize_tensor_input(input_name_to_index[k], v.shape)
        # The output names might be different in the tflite but the order is the same
        output_names = [n['name'] for n in output_details]
        interpreter.allocate_tensors()
        for k, v in feed_dict_without_port.items():
            interpreter.set_tensor(input_name_to_index[k], v)
        interpreter.invoke()
        tf_lite_output_data = [interpreter.get_tensor(output['index']) for output in output_details]

        g = process_tf_graph(None, opset=self.config.opset,
                                 input_names=list(feed_dict_without_port.keys()),
                                 output_names=output_names,
                                 target=self.config.target,
                                 tflite_path=tflite_path)
        g = optimizer.optimize_graph(g)
        onnx_feed_dict_without_port = {k.split(':')[0]: v for k, v in onnx_feed_dict.items()}
        onnx_from_tfl_output = self.run_backend(g, output_names, onnx_feed_dict_without_port, postfix="_from_tflite")

        check_value = True
        check_dtype = True
        check_shape = True
        for tf_lite_val, onnx_val in zip(tf_lite_output_data, onnx_from_tfl_output):
            if check_value:
                self.assertAllClose(tf_lite_val, onnx_val, rtol=rtol, atol=atol)
            if check_dtype:
                self.assertEqual(tf_lite_val.dtype, onnx_val.dtype)
            # why need shape checke: issue when compare [] with scalar
            # https://github.com/numpy/numpy/issues/11071
            if check_shape:
                self.assertEqual(tf_lite_val.shape, onnx_val.shape)