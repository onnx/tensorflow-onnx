<!--- SPDX-License-Identifier: Apache-2.0 -->

# tf2onnx - Convert TensorFlow models to ONNX.

| Build Type | OS | Python | Tensorflow | Onnx opset | Status |
| ---        | ---    | ---    | ---        | ---        | ---    |
| Unit Test - Basic | Linux, MacOS<sup>\*</sup>, Windows<sup>\*</sup> | 3.6, 3.7, 3.8 | 1.12-1.15, 2.1-2.4 | 7-13 | [![Build Status](https://dev.azure.com/tensorflow-onnx/tensorflow-onnx/_apis/build/status/unit_test?branchName=master)](https://dev.azure.com/tensorflow-onnx/tensorflow-onnx/_build/latest?definitionId=16&branchName=master) |
| Unit Test - Full | Linux, MacOS, Windows | 3.6, 3.7, 3.8 | 1.12-1.15, 2.1-2.4 | 7-13 | [![Build Status](https://dev.azure.com/tensorflow-onnx/tensorflow-onnx/_apis/build/status/unit_test-matrix?branchName=master)](https://dev.azure.com/tensorflow-onnx/tensorflow-onnx/_build/latest?definitionId=18&branchName=master) | |

## Supported Versions

### ONNX

tensorflow-onnx will use the ONNX version installed on your system and installs the latest ONNX version if none is found.

We support ONNX opset-6 to opset-13. By default we use opset-9 for the resulting ONNX graph since most runtimes will support opset-9.
Support for future opsets add added as they are released.

If you want the graph to be generated with a specific opset, use ```--opset``` in the command line, for example ```--opset 12```.

### TensorFlow

We support all ```tf-1.x graphs```. To keep our test matrix manageable we test tf2onnx running on top of ```tf-1.12 and up```. tf2onnx-1.5.4 was the last version that was tested all the way back to tf-1.4.

When running under tf-2.x tf2onnx will use the tensorflow V2 controlflow.

You can install tf2onnx on top of tf-1.x or tf-2.x.

### Python

We support Python ```3.6```, ```3.7``` and ```3.8```. tf2onnx-1.5.4 was the last release that supports Python 3.5.

## Status

We support many TensorFlow models. Support for Fully Connected, Convolutional and dynamic LSTM networks is mature.
A list of models that we use for testing can be found [here](tests/run_pretrained_models.yaml).

Supported RNN classes and APIs: LSTMCell, BasicLSTMCell, GRUCell, GRUBlockCell, MultiRNNCell, and user defined RNN cells inheriting rnn_cell_impl.RNNCell, used along with DropoutWrapper, BahdanauAttention, AttentionWrapper.
Check [tips](examples/rnn_tips.md) when converting RNN models.

You find a list of supported Tensorflow ops and their mapping to ONNX [here](support_status.md).

TensorFlow has broad functionality and occasionally mapping it to ONNX creates issues.
The common issues we run into we try to document here [Troubleshooting Guide](Troubleshooting.md).

## Prerequisites

### Install TensorFlow

If you don't have TensorFlow installed already, install the desired TensorFlow build, for example:

```pip install tensorflow```

or

```pip install tensorflow-gpu```

### (Optional) Install runtime

If you want to run tests, install a runtime that can run ONNX models. For example:

ONNX Runtime (available for Linux, Windows, and Mac):

```pip install onnxruntime```

For pytorch/caffe2, follow the instructions here:

```https://pytorch.org/```

We tested with pytorch/caffe2 and onnxruntime and unit tests are passing for those.

## Installation

### Install from pypi

```pip install -U tf2onnx```

### Install latest from source

```pip install git+https://github.com/onnx/tensorflow-onnx```

### Build and install latest from source (for development)

```git clone https://github.com/onnx/tensorflow-onnx```

Once dependencies are installed, from the tensorflow-onnx folder call:

```python setup.py install```

or

```python setup.py develop```

tensorflow-onnx requires onnx-1.5 or better and will install/upgrade onnx if needed.

To create a distribution:

```python setup.py bdist_wheel```

## Getting started

To get started with `tensorflow-onnx`, run the `t2onnx.convert` command, providing:

* the path to your TensorFlow model (where the model is in `saved model` format)
* a name for the ONNX output file:

```python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx```

The above command uses a default of `8` for the ONNX opset. If you need a newer opset, or want to limit your model to use an older opset then you can provide the `--opset` argument to the command. If you are unsure about which opset to use, refer to the [ONNX operator documentation](https://github.com/onnx/onnx/releases).

```python -m tf2onnx.convert --saved-model tensorflow-model-path --opset 10 --output model.onnx```

If your TensorFlow model is in a format other than `saved model`, then you need to provide the inputs and outputs of the model graph.

For `checkpoint` format:

```python -m tf2onnx.convert --checkpoint  tensorflow-model-meta-file-path --output model.onnx --inputs input0:0,input1:0 --outputs output0:0```

For `graphdef` format:

```python -m tf2onnx.convert --graphdef  tensorflow-model-graphdef-file --output model.onnx --inputs input0:0,input1:0 --outputs output0:0```

If your model is in `checkpoint` or `graphdef` format and you do not know the input and output nodes of the model, you can use the [summarize_graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) TensorFlow utility. The `summarize_graph` tool does need to be downloaded and built from source. If you have the option of going to your model provider and obtaining the model in `saved model` format, then we recommend doing so.

You find an end-to-end tutorial for ssd-mobilenet [here](tutorials/ConvertingSSDMobilenetToONNX.ipynb)

## CLI reference

```
python -m tf2onnx.convert
    --saved-model SOURCE_SAVED_MODEL_PATH |
    --checkpoint SOURCE_CHECKPOINT_METAFILE_PATH |
    --tflite SOURCE_TFLITE_PATH |
    --input | --graphdef SOURCE_GRAPHDEF_PB
    --output TARGET_ONNX_MODEL
    [--inputs GRAPH_INPUTS]
    [--outputs GRAPH_OUTPUS]
    [--inputs-as-nchw inputs_provided_as_nchw]
    [--opset OPSET]
    [--dequantize]
    [--tag TAG]
    [--signature_def SIGNATURE_DEF]
    [--concrete_function CONCRETE_FUNCTION]
    [--target TARGET]
    [--custom-ops list-of-custom-ops]
    [--fold_const]
    [--large_model]
    [--continue_on_error]
    [--verbose]
    [--output_frozen_graph]
```

### Parameters

#### --saved-model

TensorFlow model as saved_model. We expect the path to the saved_model directory.

#### --checkpoint

TensorFlow model as checkpoint. We expect the path to the .meta file.

#### --tflite

(This is experimental)

Convert a tflite model by providing a path to the .tflite file. Inputs/outputs do not need to be specified.

#### --input or --graphdef

TensorFlow model as graphdef file.

#### --output

The target onnx file path.

#### --inputs, --outputs

TensorFlow model's input/output names, which can be found with [summarize graph tool](#summarize_graph). Those names typically end with ```:0```, for example ```--inputs input0:0,input1:0```. Inputs and outputs are ***not*** needed for models in saved-model format. Some models specify placeholders with unknown ranks and dims which can not be mapped to onnx. In those cases one can add the shape after the input name inside `[]`, for example `--inputs X:0[1,28,28,3]`. Use -1 to indicate unknown dimensions.

#### --inputs-as-nchw

By default we preserve the image format of inputs (`nchw` or `nhwc`) as given in the TensorFlow model. If your hosts (for example windows) native format nchw and the model is written for nhwc, ```--inputs-as-nchw``` tensorflow-onnx will transpose the input. Doing so is convenient for the application and the converter in many cases can optimize the transpose away. For example ```--inputs input0:0,input1:0 --inputs-as-nchw input0:0``` assumes that images are passed into ```input0:0``` as nchw while the TensorFlow model given uses nhwc.

#### --ignore_default, --use_default

ONNX requires default values for graph inputs to be constant, while Tensorflow's PlaceholderWithDefault op accepts computed defaults.  To convert such models, pass a comma-separated list of node names to the ignore_default and/or use_default flags.  PlaceholderWithDefault nodes with matching names will be replaced with Placeholder or Identity ops, respectively.

#### --opset

By default we use the opset 8 to generate the graph. By specifying ```--opset``` the user can override the default to generate a graph with the desired opset. For example ```--opset 5``` would create a onnx graph that uses only ops available in opset 5. Because older opsets have in most cases fewer ops, some models might not convert on a older opset.

#### --dequantize

(This is experimental, only supported for tflite)

Produces a float32 model from a quantized tflite model. Detects ReLU and ReLU6 ops from quantization bounds.

#### --tag

Only valid with parameter `--saved_model`. Specifies the tag in the saved_model to be used. Typical value is 'serve'.

#### --signature_def

Only valid with parameter `--saved_model`. Specifies which signature to use within the specified --tag value. Typical value is 'serving_default'.

#### --concrete_function

(This is experimental, valid only for TF2.x models)

Only valid with parameter `--saved_model`. If a model contains a list of concrete functions, under the function name `__call__` (as can be viewed using the command `saved_model_cli show --all`), this parameter is a 0-based integer specifying which function in that list should be converted. This parameter takes priority over `--signature_def`, which will be ignored.

#### --large_model

(This is experimental, valid only for TF2.x models)

Only valid with parameter `--saved_model`. When set, creates a zip file containing the ONNX protobuf model and large tensor values stored externally. This allows for converting models that exceed the 2 GB protobuf limit.

#### --output_frozen_graph

Saves the frozen tensorflow graph to file.

#### --custom-ops

If a model contains ops not recognized by onnx runtime, you can tag these ops with a custom op domain so that the
runtime can still open the model. The format is a comma-separated map of tf op names to domains in the format
OpName:domain. If only an op name is provided (no colon), the default domain of `ai.onnx.converters.tensorflow`
will be used.

#### --target

Some models require special handling to run on some runtimes. In particular, the model may use unsupported data types. Workarounds are activated with ```--target TARGET```. Currently supported values are listed on this [wiki](https://github.com/onnx/tensorflow-onnx/wiki/target). If your model will be run on Windows ML, you should specify the appropriate target value.

#### --fold_const

Deprecated. Constant folding is always enabled.

### <a name="summarize_graph"></a>Tool to get Graph Inputs & Outputs

To find the inputs and outputs for the TensorFlow graph the model developer will know or you can consult TensorFlow's [summarize_graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) tool, for example:
```
summarize_graph --in_graph=tests/models/fc-layers/frozen.pb
```
### <a name="freeze_graph"></a>Tool to Freeze Graph

The TensorFlow tool to freeze the graph is [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

For example:
```
python -m tensorflow.python.tools.freeze_graph \
    --input_graph=my_checkpoint_dir/graphdef.pb \
    --input_binary=true \
    --output_node_names=output \
    --input_checkpoint=my_checkpoint_dir \
    --output_graph=tests/models/fc-layers/frozen.pb
```

## Testing
There are 2 types of tests.

### Unit test
```
python setup.py test
```

### Validate pre-trained TensorFlow models
```
python tests/run_pretrained_models.py
usage: run_pretrained_models.py [-h] [--cache CACHE] [--tests TESTS] [--backend BACKEND] [--verbose] [--debug] [--config yaml-config]

optional arguments:
  -h, --help         show this help message and exit
  --cache CACHE      pre-trained models cache dir
  --tests TESTS      tests to run
  --backend BACKEND  backend to use
  --config           yaml config file
  --verbose          verbose output, option is additive
  --opset OPSET      target opset to use
  --perf csv-file    capture performance numbers for tensorflow and onnx runtime
  --debug            dump generated graph with shape info
  --fold_const when set, TensorFlow fold_constants transformation will be applied before conversion. This will benefit features including Transpose optimization (e.g. Transpose operations introduced during tf-graph-to-onnx-graph conversion will be removed), and RNN unit conversion (for example LSTM).
```
```run_pretrained_models.py``` will run the TensorFlow model, captures the TensorFlow output and runs the same test against the specified ONNX backend after converting the model.

If the option ```--perf csv-file``` is specified, we'll capture the timeing for inferece of tensorflow and onnx runtime and write the result into the given csv file.

You call it for example with:
```
python tests/run_pretrained_models.py --backend onnxruntime --config tests/run_pretrained_models.yaml --perf perf.csv
```

#### <a name="save_pretrained_model"></a>Tool to save pre-trained model

We provide an [utility](tools/save_pretrained_model.py) to save pre-trained model along with its config.
Put `save_pretrained_model(sess, outputs, feed_inputs, save_dir, model_name)` in your last testing epoch and the pre-trained model and config will be saved under `save_dir/to_onnx`.
Please refer to the example in [tools/save_pretrained_model.py](tools/save_pretrained_model.py) for more information.
Note the minimum required Tensorflow version is r1.6.

## Python API Reference

In some cases it will be useful to convert the models from TensorFlow to ONNX from a python script. You can use the following API:

```
import tf2onnx

tf2onnx.tfonnx.process_tf_graph(tf_graph,
            continue_on_error=False, verbose=False, target=None,
            opset=None, custom_op_handlers=None,
            custom_rewriter=None, extra_opset=None,
            shape_override=None, inputs_as_nchw=None,
            input_names=None, output_names=None,
            ignore_default=None, use_default=None,
            const_node_values=None):
    """Convert tensorflow graph to onnx graph.
        Args:
            tf_graph: tensorflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            verbose: print summary stats (deprecated)
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            custom_op_handlers: dictionary of custom ops handlers
            custom_rewriter: list of custom graph rewriters
            extra_opset: list of extra opset's, for example the opset's used by custom ops
            shape_override: dict with inputs that override the shapes given by tensorflow
            inputs_as_nchw: transpose inputs in list from nchw to nchw
            input_names: list of input node names in graph, input name format as node_name:port_id
            output_names: list of output node names in graph, output name format as node_name:port_id
            ignore_default: list of node names of PlaceholderWithDefault ops to change into Placeholder ops
            use_default: list of node names of PlaceholderWithDefault ops to change into Identity ops
            const_node_values: an optional dict mapping node names to tensor values
        Return:
            onnx graph
    """
```
For example in [examples/call_converter_via_python.py]():
```
import tensorflow as tf
import tf2onnx

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["input:0"], output_names=["output:0"])
    model_proto = onnx_graph.make_model("test")
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
```

### Creating custom op mappings from python

For complex custom ops that require graph rewrites or input / attribute rewrites using the python interface to insert a custom op will be the easiest way to accomplish the task.
A dictionary of name->custom_op_handler can be passed to tf2onnx.tfonnx.process_tf_graph. If the op name is found in the graph the handler will have access to all internal structures and can rewrite that is needed. For example [examples/custom_op_via_python.py]():

```
import tensorflow as tf
import tf2onnx
from onnx import helper

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"


def print_handler(ctx, node, name, args):
    # replace tf.Print() with Identity
    #   T output = Print(T input, data, @list(type) U, @string message, @int first_n, @int summarize)
    # becomes:
    #   T output = Identity(T Input)
    node.domain = _TENSORFLOW_DOMAIN
    del node.input[1:]
    return node


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    x_ = tf.Print(x, [x], "hello")
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                 custom_op_handlers={"Print": (print_handler, ["Identity", "mode"])},
                                                 extra_opset=[helper.make_opsetid(_TENSORFLOW_DOMAIN, 1)],
                                                 input_names=["input:0"],
                                                 output_names=["output:0"])
    model_proto = onnx_graph.make_model("test")
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
```

## How tf2onnx works

The converter needs to take care of a few things:

1. Convert the protobuf format. Since the format is similar this step is straight forward.
2. TensorFlow types need to be mapped to their ONNX equivalent.
3. For many ops TensorFlow passes parameters like shapes as inputs where ONNX wants to see them as attributes. Since we use a frozen graph, the converter will fetch the input as constant, converts it to an attribute and remove the original input.
4. TensorFlow in many cases composes ops out of multiple simpler ops. The converter will need to identify the subgraph for such ops, slice the subgraph out and replace it with the ONNX equivalent. This can become fairly complex so we use a graph matching library for it. A good example of this is the tensorflow transpose op.
5. TensorFlow's default data format is NHWC where ONNX requires NCHW. The converter will insert transpose ops to deal with this.
6. There are some ops like relu6 that are not supported in ONNX but the converter can be composed out of other ONNX ops.
7. ONNX backends are new and their implementations are not complete yet. For some ops the converter generate ops with deal with issues in existing backends.

### Step 1 - start with a frozen graph

tf2onnx starts with a frozen graph. This is because of item 3 above.

### Step 2 - 1:1 conversion of the protobuf from tensorflow to onnx

tf2onnx first does a simple conversion from the TensorFlow protobuf format to the ONNX protobuf format without looking at individual ops.
We do this so we can use the ONNX graph as internal representation and write helper functions around it.
The code that does the conversion is in tensorflow_to_onnx(). tensorflow_to_onnx() will return the ONNX graph and a dictionary with shape information from TensorFlow. The shape information is helpful in some cases when processing individual ops.
The ONNX graph is wrapped in a Graph object and nodes in the graph are wrapped in a Node object to allow easier graph manipulations on the graph. All code that deals with nodes and graphs is in graph.py.

### Step 3 - rewrite subgraphs

In the next step we apply graph matching code on the graph to re-write subgraphs for ops like transpose and lstm. For an example looks at rewrite_transpose().

### Step 4 - process individual ops

In the fourth step we look at individual ops that need attention. The dictionary _OPS_MAPPING will map tensorflow op types to a method that is used to process the op. The simplest case is direct_op() where the op can be taken as is. Whenever possible we try to group ops into common processing, for example all ops that require dealing with broadcasting are mapped to broadcast_op(). For an op that composes the tensorflow op from multiple onnx ops, see relu6_op().

### Step 5 - final processing

Once all ops are converted, we need to do a topological sort since ONNX requires it. process_tf_graph() is the method that takes care of all above steps.

## Extending tf2onnx

If you like to contribute and add new conversions to tf2onnx, the process is something like:

1. See if the op fits into one of the existing mappings. If so adding it to _OPS_MAPPING is all that is needed.
2. If the new op needs extra processing, start a new mapping function.
3. If the tensorflow op is composed of multiple ops, consider using a graph re-write. While this might be a little harder initially, it works better for complex patterns.
4. Add a unit test in tests/test_backend.py. The unit tests mostly create the tensorflow graph, run it and capture the output, than convert to onnx, run against a onnx backend and compare tensorflow and onnx results.
5. If there are pre-trained models that use the new op, consider adding those to test/run_pretrained_models.py.

## License

[Apache License v2.0](LICENSE)
