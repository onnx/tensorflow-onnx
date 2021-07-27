// SPDX-License-Identifier: Apache-2.0


/* Helper script to run tfjs models. Uses custom .json representation to encode tensors for inputs/outputs
 * Example usage:
 * 
 *     node run_tfjs.js mymodel/model.json input.json output.json
 */

const tf = require('@tensorflow/tfjs');
const zlib = require("zlib");

const fs = require('fs');
const http = require('http');
const path = require('path');
const { exit } = require('process');

const [, , modelPath, inputPath, outputPath, ...args] = process.argv;

// Hide tfjs first use message complaining about lack of GPU
tf.backend().firstUse = false;

if (process.argv[2] == '--test') {
    // dtype = 'float32'|'int32'|'bool'|'complex64'|'string'

    const floatTensor = tf.tensor([1.1, 2.2, 3.3, 4.4], [2, 2], 'float32');
    const intTensor = tf.tensor([1, 2, 3, 4], [2, 2], 'int32');
    const boolTensor = tf.tensor([true, false, true, true], [2, 2], 'bool');
    const complexTensor = tf.complex([1.1, 2.2, 3.3, 4.4], [10., 20., 30., 40.]).reshape([2, 2]);
    const stringTensor = tf.tensor(['Hello world', '♦♥♠♣', '', 'Tensors'], [2, 2], 'string');

    const tensors = [floatTensor, intTensor, boolTensor, complexTensor, stringTensor];
    tensors.forEach(function (tensor) {
        const tensorEnc = tensorToJson(tensor);
        const tensorDec = tensorFromJson(tensorEnc);
        if (tensor.toString() != tensorDec.toString()) {
            console.log("Tensor:")
            tensor.print()
            console.log("Decoded tensor:")
            tensorDec.print()
            throw "Test failure"
        }
    });

    console.log("All tests pass.")
    exit(0)
}

const modelDir = path.dirname(modelPath);
const modelName = path.basename(modelPath);

const fd = fs.openSync(modelPath, 'r');
const buffer = Buffer.alloc(2);
fs.readSync(fd, buffer, 0, 2);
fs.closeSync(fd);
// Check for gzip magic number
const needsUnzip = buffer[0] == 31 && buffer[1] == 139

// tf.loadGraphModel expects a url not a local file, so we serve it on localhost 
http.createServer(function (req, res) {
    const callback = function (err, data) {
        if (err) {
            res.writeHead(404);
            res.end(JSON.stringify(err));
            return;
        }
        res.writeHead(200);
        res.end(data);
    }
    fs.readFile(modelDir + req.url, function (err, data) {
        if (err || !needsUnzip) {
            callback(err, data);
        } else {
            zlib.gunzip(data, callback);
        }
    });
}).listen(8080);

function tensorToJson(tensor) {
    if (tensor.dtype != 'string') {
        const data = tensor.dataSync()
        const byteArray = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        const dataEnc = Buffer.from(byteArray).toString('base64');
        return {
            dataEnc: dataEnc,
            shape: tensor.shape,
            dtype: tensor.dtype
        };
    }

    return {
        data: tensor.dataSync(),
        shape: tensor.shape,
        dtype: tensor.dtype
    };
}

function getTypedArrayConstructor(dtype) {
    if (dtype == 'complex64') {
        return Float32Array;
    }
    return tf.util.getTypedArrayFromDType(dtype).constructor;
}

function tensorFromJson(json) {
    let data = json.data;
    if (data == undefined) {
        const arrayType = getTypedArrayConstructor(json.dtype);
        data = new arrayType(new Uint8Array(Buffer.from(json.dataEnc, 'base64')).buffer);
    }
    if (json.dtype == 'complex64') {
        const floatTensor = tf.tensor(data, [data.length], 'float32').reshape([-1, 2]).transpose();
        return tf.complex(floatTensor.gather(0), floatTensor.gather(1)).reshape(json.shape);
    }
    return tf.tensor(data, json.shape, json.dtype);
}

function inputFromJson(json) {
    // Input can be a tensor, list of tensors, or mapping of input names to tensors.
    if (Array.isArray(json)) {
        return json.map(tensorFromJson);
    }
    const keys = Object.keys(json);
    if (keys.length == 0 || json[keys[0]].dtype != undefined) {
        const result = {};
        keys.forEach(k => { result[k] = tensorFromJson(json[k]); });
        return result;
    }
    return tensorFromJson(json);
}

function outputToJson(out) {
    // Output can be a tensor, list of tensors, or mapping of output names to tensors.
    if (Array.isArray(out)) {
        return out.map(tensorToJson);
    }
    if (out instanceof tf.Tensor) {
        return tensorToJson(out);
    }
    const result = {};
    Object.keys(out).forEach(k => { result[k] = tensorToJson(out[k]); });
    return result;
}

async function main() {
    const model = await tf.loadGraphModel('http://localhost:8080/' + modelName);
    const inputString = fs.readFileSync(inputPath, 'utf8');
    const inputJson = JSON.parse(inputString);
    const input = inputFromJson(inputJson);
    let outputs = null;
    if (args && args[0] == '--outputs') {
        outputs = args[1].split(',');
    }
    const output = await model.executeAsync(input, outputs);

    const outputJson = outputToJson(output);
    const outputString = JSON.stringify(outputJson);
    fs.writeFileSync(outputPath, outputString, 'utf8');
}

main().then(() => exit(0)).catch((err) => { console.error(err); exit(1) })
