# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tflite_utils - utilities for parsing tflite files into onnx graph
"""

import collections
import importlib

from onnx import helper, onnx_pb, numpy_helper
from tensorflow.core.framework import types_pb2, tensor_pb2
from tensorflow.python.framework import tensor_util
from tflite.TensorType import TensorType as TFLiteTensorType
from tflite.Model import Model


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
    TFLiteTensorType.UINT64: onnx_pb.TensorProto.UINT64,
}


TFLITE_TO_TF_DTYPE = {
    TFLiteTensorType.FLOAT32: types_pb2.DT_FLOAT,
    TFLiteTensorType.FLOAT16: types_pb2.DT_HALF,
    TFLiteTensorType.INT32: types_pb2.DT_INT32,
    TFLiteTensorType.UINT8: types_pb2.DT_UINT8,
    TFLiteTensorType.INT64: types_pb2.DT_INT64,
    TFLiteTensorType.STRING: types_pb2.DT_STRING,
    TFLiteTensorType.BOOL: types_pb2.DT_BOOL,
    TFLiteTensorType.INT16: types_pb2.DT_INT16,
    TFLiteTensorType.COMPLEX64: types_pb2.DT_COMPLEX64,
    TFLiteTensorType.INT8: types_pb2.DT_INT8,
    TFLiteTensorType.FLOAT64: types_pb2.DT_DOUBLE,
    TFLiteTensorType.COMPLEX128: types_pb2.DT_COMPLEX128,
    TFLiteTensorType.UINT64: types_pb2.DT_UINT64,
}


def map_tflite_dtype_to_onnx(dtype):
    return TFLITE_TO_ONNX_DTYPE[dtype]


def map_tflite_dtype_to_tf(dtype):
    return TFLITE_TO_TF_DTYPE[dtype]


# The tflite schema uses snake case, but the python bindings use proper case
def snake_to_proper_case(name):
    return ''.join(n.capitalize() for n in name.split('_'))


def proper_to_snake_case(name):
    res = ''
    for c in name:
        if c.isupper() and res:
            res += '_'
        res += c.lower()
    return res

# Pulled from the tflite schema.fbs file. Needed to decode enum numbers into strings.
NODE_ATTR_NAME_TO_ENUM_TYPE = {
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
NODE_ATTR_NAME_TO_ENUM_TYPE = {snake_to_proper_case(key): value for key, value in NODE_ATTR_NAME_TO_ENUM_TYPE.items()}

# Pulled from the tflite schema.fbs file.
FUNCTION_ATTRS = ['then_subgraph_index', 'else_subgraph_index', 'cond_subgraph_index',
                  'body_subgraph_index', 'subgraph']
FUNCTION_ATTRS = [snake_to_proper_case(attr) for attr in FUNCTION_ATTRS]


enum_cache = {}
def lookup_enum(idx, enum_name):
    """Given the name of a tflite enum class and an index, return a string with the name of the enum value"""
    if enum_name == 'TensorType':
        return map_tflite_dtype_to_onnx(idx)
    if enum_name in enum_cache:
        return enum_cache[enum_name][idx]
    module = importlib.import_module('tflite.' + enum_name)
    enum_class = getattr(module, enum_name)
    idx_to_name = {value: key for key, value in enum_class.__dict__.items() if not key.startswith('_')}
    enum_cache[enum_name] = idx_to_name
    return idx_to_name[idx]


def get_options_class(name):
    """Each tflite optype has a flatbuffer Options class (ex: AddOptions). Returns the options class given its name."""
    if name == "NONE":
        return None
    module = importlib.import_module('tflite.' + name)
    return getattr(module, name)


def read_tflite_model(tflite_path):
    """
    Given the path to a tflite model, returns tuple (tflite_graphs, opcodes_map, model)
    Pass these to parse_tflite_graph
    """
    with open(tflite_path, 'rb') as f:
        buf = f.read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    # To save space, each op in the model indicates its opcode as an index into the model's opcode map.
    opcodes_map = {}
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        # TFlite ran out of opcodes since they only used a byte. Old models store opcodes in DeprecatedBuiltinCode.
        # New models put PLACEHOLDER_FOR_GREATER_OP_CODES in this field to signify that BuiltinCode should be used.
        code = lookup_enum(op_code.DeprecatedBuiltinCode(), 'BuiltinOperator')
        if code == 'PLACEHOLDER_FOR_GREATER_OP_CODES':
            code = lookup_enum(op_code.BuiltinCode(), 'BuiltinOperator')
        opcodes_map[i] = code
    tflite_graphs = [model.Subgraphs(i) for i in range(model.SubgraphsLength())]
    return tflite_graphs, opcodes_map, model


def parse_tflite_graph(tflite_g, opcodes_map, model, input_prefix=''):
    """
    Returns a Graph object along with some op count stats. All tflite op types are prefixed with "TFL_".
    Names of graph inputs are optionally prefixed with a string to prevent name conflicts in subgraphs.
    Quantizatized tensors are surrounded with quantize/dequantize ops
    """
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}
    tensor_names = {}
    # Map tensor name to tflite Tensor object so we can fetch quantization info as needed
    name_to_tensor = {}
    # If a node takes a quantized tensor as input, we must add a dequantize op after it.
    # Store a mapping so we only need to make at most one dequantize op per tensor.
    tensor_name_to_dequant_output = {}

    # tflite uses generic names (arg0, arg1, etc.) for inputs but full names for other tensors, so
    # prefixing just the inputs should be fine. Other tensors are prefixed when we do inlining.
    input_indices = {tflite_g.Inputs(i) for i in range(tflite_g.InputsLength())}

    for i in range(tflite_g.TensorsLength()):
        tensor = tflite_g.Tensors(i)
        name = tensor.Name().decode()
        if i in input_indices:
            name = input_prefix + name
        tensor_names[i] = name
        name_to_tensor[name] = tensor

        if tensor.ShapeIsNone():
            output_shapes[name] = None
        elif tensor.ShapeSignatureIsNone():
            # The shape signature uses -1 to signify unknown dims. Old models don't have this and use Shape instead.
            output_shapes[name] = tensor.ShapeAsNumpy().tolist()
        else:
            output_shapes[name] = tensor.ShapeSignatureAsNumpy().tolist()
        buf = model.Buffers(tensor.Buffer())
        dtypes[name] = map_tflite_dtype_to_onnx(tensor.Type())
        if not buf.DataIsNone():
            # For const values we use TF to decode the binary data from the buffer
            t = tensor_pb2.TensorProto()
            t.tensor_content = buf.DataAsNumpy().tobytes()
            if output_shapes[name] is None:
                output_shapes[name] = []
            for d in output_shapes[name]:
                t.tensor_shape.dim.add().size = d
            t.dtype = map_tflite_dtype_to_tf(tensor.Type())
            np_data = tensor_util.MakeNdarray(t)
            onnx_tensor = numpy_helper.from_array(np_data, name=name)
            onnx_node = helper.make_node("Const", [], outputs=[name], name=name, value=onnx_tensor)
            onnx_nodes.append(onnx_node)
            op_cnt["Const"] += 1

    def get_dequant(tensor_name):
        """Creates a dequantize op for the provided tensor if needed and returns the output of the op, or
        the original tensor name if no dequantization is needed"""
        quant = name_to_tensor[tensor_name].Quantization()
        if quant is None or quant.ScaleIsNone() or quant.ZeroPointIsNone():
            return tensor_name
        if tensor_name in tensor_name_to_dequant_output:
            return tensor_name_to_dequant_output[tensor_name]
        dequant_name = tensor_name + "_dequant"
        attr = {}
        attr['scale'] = quant.ScaleAsNumpy().tolist()
        attr['zero_point'] = quant.ZeroPointAsNumpy().tolist()
        attr['quantized_dimension'] = quant.QuantizedDimension()
        onnx_node = helper.make_node("TFL_DEQUANTIZE", [tensor_name], [dequant_name], name=dequant_name, **attr)
        onnx_nodes.append(onnx_node)
        tensor_name_to_dequant_output[tensor_name] = dequant_name
        output_shapes[dequant_name] = output_shapes[tensor_name].copy()
        dtypes[dequant_name] = onnx_pb.TensorProto.FLOAT
        return dequant_name

    def get_prequant(tensor_name):
        """Called by nodes with the name of the tensor they must output.
        If the output is supposed to be quantized, creates a Quantize op outputting the tensor.
        Returns the name that should be used for the "prequantized" tensor, or the original tensor if no quantization
        is needed"""
        quant = name_to_tensor[tensor_name].Quantization()
        if quant is None or quant.ScaleIsNone() or quant.ZeroPointIsNone():
            return tensor_name
        prequant_name = tensor_name + "_prequant"
        quantize_name = tensor_name + "_quantize"
        attr = {}
        attr['scale'] = quant.ScaleAsNumpy().tolist()
        attr['zero_point'] = quant.ZeroPointAsNumpy().tolist()
        attr['quantized_dimension'] = quant.QuantizedDimension()
        onnx_node = helper.make_node("TFL_QUANTIZE", [prequant_name], [tensor_name], name=quantize_name, **attr)
        onnx_nodes.append(onnx_node)
        output_shapes[prequant_name] = output_shapes[tensor_name].copy()
        dtypes[prequant_name] = onnx_pb.TensorProto.FLOAT
        return prequant_name

    for i in range(tflite_g.OperatorsLength()):
        op = tflite_g.Operators(i)
        optype = opcodes_map[op.OpcodeIndex()]
        op_cnt[optype] += 1
        attr = {}
        options_type_name = lookup_enum(op.BuiltinOptionsType(), 'BuiltinOptions')
        option_class = get_options_class(options_type_name)
        wants_dequantized_input = True
        has_prequantized_output = True
        if optype == 'QUANTIZE':
            out_tensor = tflite_g.Tensors(op.Outputs(0))
            quant = out_tensor.Quantization()
            has_prequantized_output = False
            if quant is not None and not quant.ScaleIsNone() and not quant.ZeroPointIsNone():
                attr['scale'] = quant.ScaleAsNumpy().tolist()
                attr['zero_point'] = quant.ZeroPointAsNumpy().tolist()
                attr['quantized_dimension'] = quant.QuantizedDimension()
        elif optype == 'DEQUANTIZE':
            in_tensor = tflite_g.Tensors(op.Inputs(0))
            quant = in_tensor.Quantization()
            wants_dequantized_input = False
            if quant is not None and not quant.ScaleIsNone() and not quant.ZeroPointIsNone():
                attr['scale'] = quant.ScaleAsNumpy().tolist()
                attr['zero_point'] = quant.ZeroPointAsNumpy().tolist()
                attr['quantized_dimension'] = quant.QuantizedDimension()
        if option_class is not None:
            options = option_class()
            options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            # All flatbuffer objects have these properties.
            block_list = [options_type_name + 'BufferHasIdentifier', 'Init', 'GetRootAs' + options_type_name]
            # The rest of the properties of the options class provide its attribute names
            attr_names = {opt for opt in dir(options) if not opt.startswith('_') and opt not in block_list}
            for a in list(attr_names):
                # Flatbufffer list properties have 3 functions: *Length, *IsNone, and *AsNumpy
                if a + 'Length' in attr_names:
                    attr_names.remove(a + 'Length')
                    attr_names.remove(a + 'IsNone')
                    attr_names.remove(a)
            for a in attr_names:
                if a.endswith('AsNumpy'):
                    value = getattr(options, a)().tolist()
                    a = a[:-len('AsNumpy')]
                else:
                    # For enums we use a string with the value name, not enum index
                    value = getattr(options, a)()
                    if a in NODE_ATTR_NAME_TO_ENUM_TYPE:
                        value = lookup_enum(value, NODE_ATTR_NAME_TO_ENUM_TYPE[a])
                    elif a in FUNCTION_ATTRS:
                        value = model.Subgraphs(value).Name().decode()
                attr_cnt[a] += 1
                attr[proper_to_snake_case(a)] = value
        input_names = [tensor_names[op.Inputs(i)] for i in range(op.InputsLength()) if op.Inputs(i) != -1]
        if wants_dequantized_input:
            input_names = [get_dequant(inp) for inp in input_names]
        output_names = [tensor_names[op.Outputs(i)] for i in range(op.OutputsLength()) if op.Outputs(i) != -1]
        if has_prequantized_output:
            output_names = [get_prequant(out) for out in output_names]
        onnx_node = helper.make_node("TFL_" + optype, input_names, output_names, name=output_names[0], **attr)
        onnx_nodes.append(onnx_node)

    inputs = [tensor_names[tflite_g.Inputs(i)] for i in range(tflite_g.InputsLength())]
    outputs = [tensor_names[tflite_g.Outputs(i)] for i in range(tflite_g.OutputsLength())]
    # TODO: Allow input/outputs to be overridden

    for inp in inputs:
        onnx_node = helper.make_node("Placeholder", [], outputs=[inp], name=inp)
        onnx_nodes.append(onnx_node)

    graph_name = (tflite_g.Name() or b'tflite graph').decode()
    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, inputs, outputs, graph_name
