import os
import sys
import subprocess
import numpy as np
import json
import base64

def numpy_to_json(np_arr):
    # TFJS only has types float32, int32, bool, string and complex64
    dtype_map = {
        'b': 'bool',
        'i': 'int32',
        'u': 'int32',
        'S': 'string',
        'O': 'string',
        'U': 'string',
        'c': 'complex64',
        'f': 'float32',
    }
    dtype = dtype_map[np_arr.dtype.kind]
    result = {
        'shape': list(np_arr.shape),
        'dtype': dtype,
    }
    if dtype == 'string':
        result['data'] = np_arr.flatten().tolist()
    else:
        result['dataEnc'] = base64.encodebytes(np_arr.astype(dtype).tobytes()).decode()
    return result

def json_to_numpy(obj):
    dtype = obj['dtype']
    if dtype == 'string':
        dtype = 'str'
    if 'data' in obj:
        return np.array(obj['data'], dtype).reshape(obj['shape'])
    data_bytes = base64.decodebytes(obj['dataEnc'].encode())
    return np.frombuffer(data_bytes, dtype).reshape(obj['shape'])

def input_to_json(input):
    if isinstance(input, list):
        return [numpy_to_json(arr) for arr in input]
    if isinstance(input, dict):
        return {k: numpy_to_json(v) for k, v in input.items()}
    return numpy_to_json(input)

def json_to_output(obj):
    if isinstance(obj, list):
        return [json_to_numpy(x) for x in obj]
    if all(isinstance(x, dict) for x in obj.values()):
        return {k: json_to_numpy(v) for k, v in obj.itms()}
    return [json_to_numpy(obj)]

def test():
    float_array = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
    int_array = np.array([[1, 2], [3, 4]], np.int32)
    bool_array = np.array([[True, False], [True, True]], np.bool)
    string_array = np.array([['Hello world', ''], ['test', 'Tensor']], np.str)
    complex_array = np.array([[1 + 0.1j, 2 + 0.2j], [3 + 0.3j, 4 + 0.4j]], np.complex64)

    arrays = [float_array, int_array, bool_array, string_array, complex_array]
    for arr in arrays:
        array_enc = numpy_to_json(arr)
        print(array_enc)
        array_dec = json_to_numpy(array_enc)
        np.testing.assert_equal(arr, array_dec)

    print("Tests pass")

def run_tfjs(tfjs_path, inputs):
    script_path = os.path.join(os.path.dirname(__file__), 'run_tfjs.js')
    working_dir = os.path.dirname(os.path.dirname(tfjs_path))
    input_path = os.path.join(working_dir, 'input.json')
    output_path = os.path.join(working_dir, 'output.json')
    stderr_path = os.path.join(working_dir, 'stderr.txt')

    with open(input_path, 'wt') as f:
        json.dump(input_to_json(inputs), f)

    with open(stderr_path, 'wb') as f:
        result = subprocess.run(['node', script_path, tfjs_path, input_path, output_path], stderr=f)
    if result.returncode != 0:
        with open(stderr_path, 'rt') as f:
            err = f.read()
        raise RuntimeError("Failed to run tfjs model: " + err)

    with open(output_path, 'rt', encoding='utf8') as f:
        result = json_to_output(json.load(f))

    return result
