# SPDX-License-Identifier: Apache-2.0

"""
Generates the files in tf2onnx/tflite used for parsing tflite flatbuffer
WARNING: this script will overwrite all files in tf2onnx/tflite
Before running, download the flatc executable from https://github.com/google/flatbuffers/releases and add it to PATH
This script only tested on Windows
"""

import os
import subprocess
import tempfile
import wget

SCHEMA_URL = "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/schema/schema.fbs"

FILE_HEADER = "# SPDX-License-Identifier: Apache-2.0\n\n"

def main():
    tmpdir = os.path.join(tempfile.gettempdir(), "tflite_flatbuffer")
    os.makedirs(tmpdir, exist_ok=True)
    repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dstpath = os.path.join(repodir, "tf2onnx", "tflite")
    os.makedirs(dstpath, exist_ok=True)
    # Remove existing flatbuffer bindings
    for file in os.listdir(dstpath):
        os.remove(os.path.join(dstpath, file))
    schema_path = os.path.join(tmpdir, "schema.fbs")
    # Download schema file
    if os.path.exists(schema_path):
        os.remove(schema_path)
    wget.download(SCHEMA_URL, schema_path)
    print()
    # Generate flatbuffer code
    subprocess.call(["flatc", "-p", "-o", tmpdir, schema_path])
    tmp_result_path = os.path.join(tmpdir, "tflite")
    for file in os.listdir(tmp_result_path):
        with open(os.path.join(tmp_result_path, file), "rt") as f:
            content = f.read()
        content = FILE_HEADER + content.replace("from tflite.", "from tf2onnx.tflite.")
        with open(os.path.join(dstpath, file), "wt") as f:
            f.write(content)
        print("Generated", file)

main()
