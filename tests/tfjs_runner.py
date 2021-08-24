# SPDX-License-Identifier: Apache-2.0


"""Contains tools for running tfjs models. Requires nodejs."""

import os
import subprocess
import json
import base64

import numpy as np


def numpy_to_json(np_arr):
    """Encodes a numpy array to a json-serializable dict"""
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
        # This is faster for JSON to parse and can represent inf/nan values
        result['dataEnc'] = base64.encodebytes(np_arr.astype(dtype).tobytes()).decode()
    return result


def json_to_numpy(obj):
    """Decodes a dict from json into a numpy array"""
    dtype = obj['dtype']
    if dtype == 'string':
        dtype = 'str'
    if 'data' in obj:
        return np.array(obj['data'], dtype).reshape(obj['shape'])
    data_bytes = base64.decodebytes(obj['dataEnc'].encode())
    return np.frombuffer(data_bytes, dtype).reshape(obj['shape'])


def inputs_to_json(inputs):
    """Encodes a dict, list or single numpy array into a corresponding json representation understood by run_tfjs.js"""
    if isinstance(inputs, list):
        return [numpy_to_json(arr) for arr in inputs]
    if isinstance(inputs, dict):
        return {k: numpy_to_json(v) for k, v in inputs.items()}
    return numpy_to_json(inputs)


def json_to_output(obj):
    """Given an object representing an output from run_tfjs.js, returns a list of numpy arrays (representing the
    output from the model). NOTE: the output of this function is always a list of arrays."""
    if isinstance(obj, list):
        return [json_to_numpy(x) for x in obj]
    if all(isinstance(x, dict) for x in obj.values()):
        return [json_to_numpy(v) for v in obj.values()]
    return [json_to_numpy(obj)]


def run_tfjs(tfjs_path, inputs, outputs=None):
    """
    Given the path to the model.json of a tfjs model, a dict mapping input names to numpy arrays, and a working
    directory, runs the model on the inputs and returns the resulting arrays or raises a RuntimeException. Calls
    run_tfjs.js using nodejs.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'run_tfjs.js')
    working_dir = os.path.dirname(os.path.dirname(tfjs_path))
    input_path = os.path.join(working_dir, 'input.json')
    output_path = os.path.join(working_dir, 'output.json')
    stderr_path = os.path.join(working_dir, 'stderr.txt')

    command = ['node', script_path, tfjs_path, input_path, output_path]
    if outputs is not None:
        command.extend(['--outputs', ','.join(outputs)])

    with open(input_path, 'wt') as f:
        json.dump(inputs_to_json(inputs), f)

    with open(stderr_path, 'wb') as f:
        result = subprocess.run(command, stderr=f, check=False)
    if result.returncode != 0:
        with open(stderr_path, 'rt') as f:
            err = f.read()
        raise RuntimeError("Failed to run tfjs model: " + err)

    with open(output_path, 'rt', encoding='utf8') as f:
        result = json_to_output(json.load(f))

    return result
