import numpy as np
import tensorflow as tf
from onnx import helper, onnx_pb, numpy_helper

from tflite.TensorType import TensorType as TFLiteTensorType

TFLITE_TO_ONNX_DTYPE = {
    TFLiteTensorType.FLOAT32: onnx_pb.TensorProto.FLOAT,
    TFLiteTensorType.FLOAT16: onnx_pb.TensorProto.FLOAT16,
    TFLiteTensorType.INT32: onnx_pb.TensorProto.INT32,
    TFLiteTensorType.UINT8: onnx_pb.TensorProto.UINT8,
    TFLiteTensorType.INT64: onnx_pb.TensorProto.INT64,
    TFLiteTensorType.STRING: onnx_pb.TensorProto.STRING,
    TFLiteTensorType.BOOL: onnx_pb.TensorProto.BOOL,
    TFLiteTensorType.INT16: onnx_pb.TensorProto.INT16,
    TFLiteTensorType.COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    TFLiteTensorType.INT8: onnx_pb.TensorProto.INT8,
    TFLiteTensorType.FLOAT64: onnx_pb.TensorProto.DOUBLE,
    TFLiteTensorType.COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
}

def snake_to_proper_case(name):
    return ''.join(n.capitalize() for n in name.split('_'))

def proper_to_snake_case(name):
    res = ''
    for c in name:
        if c.isupper() and res:
            res += '_'
        res += c.lower()
    return res

OPTION_TO_ENUM_TYPE = {
    'fused_activation_function': 'ActivationFunctionType',
    'padding': 'Padding',
    'type': 'LSHProjectionType',
    'weights_format': 'FullyConnectedOptionsWeightsFormat',
    'kernel_type': 'LSTMKernelType',
    'combiner': 'CombinerType',
    'in_data_type': 'TensorType',
    'out_data_type': 'TensorType',
    'output_type': 'TensorType',
    'out_type': 'TensorType',
    'mode': 'MirrorPadMode',
    'idx_out_type': 'TensorType',
}
OPTION_TO_ENUM_TYPE = {snake_to_proper_case(key): value for key, value in OPTION_TO_ENUM_TYPE.items()}

enum_cache = {}
def lookup_enum(idx, enum_name):
    if enum_name in enum_cache:
        return enum_cache[enum_name][idx]
    import importlib
    module = importlib.import_module('tflite.' + enum_name)
    enum_class = getattr(module, enum_name)
    idx_to_name = {value: key for key, value in enum_class.__dict__.items() if not key.startswith('_')}
    enum_cache[enum_name] = idx_to_name
    return idx_to_name[idx]

def tflite_graph_to_onnx(tflite_g, opcodes, option_classes):
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}
    tensor_names = {}

    for i in range(tflite_g.TensorsLength()):
        tensor = tflite_g.Tensors(i)
        name = tensor.Name()
        tensor_names[i] = name
        output_shapes[name] = tensor.ShapeAsNumpy()
        dtypes[name] = TFLITE_TO_ONNX_DTYPE[tensor.Type()]

    for i in range(tflite_g.OperatorsLength()):
        op = tflite_g.Operators(i)
        optype = opcodes[op.OpcodeIndex()]
        attr = {}
        option_class = option_classes[op.BuiltinOptionsType()]
        if option_class is not None:
            options = option_class()
            options_type_name = lookup_enum(op.BuiltinOptionsType(), 'BuiltinOptions')
            options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            block_list = [options_type_name + 'BufferHasIdentifier', 'Init', 'GetRootAs' + options_type_name]
            attr_names = {opt for opt in dir(options) if not opt.startswith('_') and opt not in block_list}
            for a in list(attr_names):
                if a + 'Length' in attr_names:
                    attr_names.remove(a + 'Length')
                    attr_names.remove(a + 'IsNone')
                    attr_names.remove(a)
            for a in attr_names:
                if a.endswith('AsNumpy'):
                    value = getattr(options, a)()
                    a = a[:-len('AsNumpy')]
                else:
                    value = getattr(options, a)()
                    if a in OPTION_TO_ENUM_TYPE:
                        value = lookup_enum(value, OPTION_TO_ENUM_TYPE[a])
                attr[proper_to_snake_case(a)] = value
        input_names = [tensor_names[op.Inputs(i)] for i in range(op.InputsLength())]
        output_names = [tensor_names[op.Outputs(i)] for i in range(op.OutputsLength())]
        onnx_node = helper.make_node("TFL_" + optype, input_names, output_names, name=output_names[0], **attr)
        onnx_nodes.append(onnx_node)

    return onnx_nodes, output_shapes, dtypes

def main():
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.BuiltinOptions import BuiltinOptions
    buf = open(r"model.tflite", 'rb').read()
    buf = bytearray(buf)
    m = Model.GetRootAsModel(buf, 0)
    import importlib
    def get_options(name):
        if name == "NONE":
            return None
        module = importlib.import_module('tflite.' + name)
        return getattr(module, name)
    lookup_enum(0, 'BuiltinOptions')
    option_classes = {key: get_options(value) for key, value in enum_cache['BuiltinOptions'].items()}
    opcodes = {}
    for i in range(m.OperatorCodesLength()):
        code = m.OperatorCodes(i)
        opcodes[i] = lookup_enum(code.BuiltinCode(), 'BuiltinOperator')
    onnx_graphs = [tflite_graph_to_onnx(m.Subgraphs(i), opcodes, option_classes) for i in range(m.SubgraphsLength())]
    pass


def test_inference():
    interpreter = tf.lite.Interpreter(model_path="C:/Users/tomwi/OneDrive - Microsoft/ONNX/tensorflow-onnx/mobilenet_v1_1.0_224_quant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

#main()