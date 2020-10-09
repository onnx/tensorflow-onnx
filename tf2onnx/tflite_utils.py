import numpy as np
import tensorflow as tf
from onnx import helper, onnx_pb, numpy_helper
from tensorflow.core.framework import types_pb2, tensor_pb2
from tensorflow.python.framework import tensor_util
import collections

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

TFLITE_TO_TF_DTYPE = {
    TFLiteTensorType.FLOAT32: types_pb2.DT_FLOAT,
    TFLiteTensorType.FLOAT16: types_pb2.DT_HALF,
    TFLiteTensorType.INT32: types_pb2.DT_INT32,
    TFLiteTensorType.UINT8: types_pb2.DT_INT8,
    TFLiteTensorType.INT64: types_pb2.DT_INT64,
    TFLiteTensorType.STRING: types_pb2.DT_STRING,
    TFLiteTensorType.BOOL: types_pb2.DT_BOOL,
    TFLiteTensorType.INT16: types_pb2.DT_INT16,
    TFLiteTensorType.COMPLEX64: types_pb2.DT_COMPLEX64,
    TFLiteTensorType.INT8: types_pb2.DT_INT8,
    TFLiteTensorType.FLOAT64: types_pb2.DT_DOUBLE,
    TFLiteTensorType.COMPLEX128: types_pb2.DT_COMPLEX128,
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

FUNCTION_ATTRS = ['then_subgraph_index', 'else_subgraph_index', 'cond_subgraph_index', 'body_subgraph_index', 'subgraph']
FUNCTION_ATTRS = [snake_to_proper_case(attr) for attr in FUNCTION_ATTRS]

enum_cache = {}
def lookup_enum(idx, enum_name):
    if enum_name == 'TensorType':
        return TFLITE_TO_ONNX_DTYPE[idx]
    if enum_name in enum_cache:
        return enum_cache[enum_name][idx]
    import importlib
    module = importlib.import_module('tflite.' + enum_name)
    enum_class = getattr(module, enum_name)
    idx_to_name = {value: key for key, value in enum_class.__dict__.items() if not key.startswith('_')}
    enum_cache[enum_name] = idx_to_name
    return idx_to_name[idx]

import importlib
def get_options_class(name):
    if name == "NONE":
        return None
    module = importlib.import_module('tflite.' + name)
    return getattr(module, name)

def read_tflite_model(tflite_path):
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.BuiltinOptions import BuiltinOptions
    buf = open(tflite_path, 'rb').read()
    buf = bytearray(buf)
    m = Model.GetRootAsModel(buf, 0)
    opcodes = {}
    for i in range(m.OperatorCodesLength()):
        code = m.OperatorCodes(i)
        opcodes[i] = lookup_enum(code.BuiltinCode(), 'BuiltinOperator')
    tflite_graphs = [m.Subgraphs(i) for i in range(m.SubgraphsLength())]
    return tflite_graphs, opcodes, m

def tflite_graph_to_onnx(tflite_g, opcodes, model):
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}
    tensor_names = {}

    for i in range(tflite_g.TensorsLength()):
        tensor = tflite_g.Tensors(i)
        name = tensor.Name().decode()
        tensor_names[i] = name

        output_shapes[name] = tensor.ShapeAsNumpy().tolist()
        buf = model.Buffers(tensor.Buffer())
        dtypes[name] = TFLITE_TO_ONNX_DTYPE[tensor.Type()]
        if not buf.DataIsNone():
            t = tensor_pb2.TensorProto()
            t.tensor_content = buf.DataAsNumpy().tobytes()
            for d in output_shapes[name]:
                t.tensor_shape.dim.add().size = d
            t.dtype = TFLITE_TO_TF_DTYPE[tensor.Type()]
            np_data = tensor_util.MakeNdarray(t)
            onnx_tensor = numpy_helper.from_array(np_data, name=name)
            onnx_node = helper.make_node("Const", [], outputs=[name], name=name, value=onnx_tensor)
            onnx_nodes.append(onnx_node)
            op_cnt["Const"] += 1

    for i in range(tflite_g.OperatorsLength()):
        op = tflite_g.Operators(i)
        optype = opcodes[op.OpcodeIndex()]
        op_cnt[optype] += 1
        attr = {}
        options_type_name = lookup_enum(op.BuiltinOptionsType(), 'BuiltinOptions')
        option_class = get_options_class(options_type_name)
        if option_class is not None:
            options = option_class()
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
                    elif a in FUNCTION_ATTRS:
                        value = model.Subgraphs(value).Name().decode()
                attr_cnt[a] += 1
                attr[proper_to_snake_case(a)] = value
        input_names = [tensor_names[op.Inputs(i)] for i in range(op.InputsLength())]
        output_names = [tensor_names[op.Outputs(i)] for i in range(op.OutputsLength())]
        onnx_node = helper.make_node("TFL_" + optype, input_names, output_names, name=output_names[0], **attr)
        onnx_nodes.append(onnx_node)

    inputs = [tflite_g.Tensors(tflite_g.Inputs(i)).Name().decode() for i in range(tflite_g.InputsLength())]
    outputs = [tflite_g.Tensors(tflite_g.Outputs(i)).Name().decode() for i in range(tflite_g.OutputsLength())]

    for inp in inputs:
        onnx_node = helper.make_node("Placeholder", [], outputs=[inp], name=inp)
        onnx_nodes.append(onnx_node)

    graph_name = tflite_g.Name().decode()
    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, inputs, outputs, graph_name

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