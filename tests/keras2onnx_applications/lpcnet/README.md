<!--- SPDX-License-Identifier: Apache-2.0 -->

# Introduction
This tool converts the lpcnet model to onnx.
To run this code, we need first install the original lpcnet model from <https://github.com/mozilla/LPCNet/>.
Note that lpcnet is not a package, so please add its directory to the path.
Then run
```
python convert_lpcnet_to_onnx.py [model_file]
```
model_file is the model with trained weights, it is a *.h5 file.
