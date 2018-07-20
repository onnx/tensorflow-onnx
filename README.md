tf2onnx - convert TensorFlow models to ONNX models.
========

Tf2onnx converts a TensorFlow graph to an ONNX graph.

Tf2onnx is in its early development. Mileage will vary since TensorFlow supports ~4 times the operations that the current ONNX version supports. But standard models seem to be using mostly ops that ONNX does support.

| Linux |
|-------|
| [![Build Status](https://travis-ci.org/onnx/tensorflow-onnx.svg?branch=master)](https://travis-ci.org/onnx/tensorflow-onnx)

# Status
Basic net and conv nets should work. A list of models that pass tests can be found [here](tests/run_pretrained_models.yaml)

# Prerequisites

## Install TensorFlow
If you don't have tensorflow installed already, install the desired tensorflow build, for example:
```
pip install tensorflow
or
pip install tensorflow-gpu
```
## Install Caffe2 [**Optional**]
If you want to run unit tests against the Caffe2 onnx backend, build and install Caffe2 following the instructions here: ```
https://caffe2.ai/```

## Python Version
We tested with tensorflow 1.5,1.6,1.7,1.8 and anaconda **3.5,3.6**.

# Installation

Once dependencies are installed, from the tensorflow-onnx folder call:

```
python setup.py install
or 
python setup.py develop
```
tensorflow-onnx requires onnx-1.2.2 or better and will install/upgrade onnx if needed.

To create a distribution:
```
python setup.py sdist
```

# Usage

To convert a TensorFlow model, tf2onnx expects a ```frozen TensorFlow graph``` and the user needs to specify inputs and outputs for the graph by passing the input and output
names with ```--inputs INPUTS``` and ```--outputs OUTPUTS```. 

Usage command:
```
python -m tf2onnx.convert --input SOURCE_FROZEN_GRAPH_PB\
    --inputs SOURCE_GRAPH_INPUTS\
    --outputs SOURCE_GRAPH_OUTPUS\
    [--output TARGET_ONNX_GRAPH]\
    [--target TARGET]\
    [--continue_on_error]\
    [--verbose]\
    [--custom-ops list-of-custom-ops]\
    [--opset OPSET]
```

Parameters:
- input: frozen TensorFlow graph, which can be created with the [freeze graph tool](#freeze_graph).
- output: the target onnx file path.
- inputs/outputs: Tensorflow graph's input/output names, which can be found with [summarize graph tool](#summarize_graph).
- target: There are different onnx versions and workarounds for runtimes that can be set with ```--target TARGET```. 
- opset: by default we uses the newest opset installed with the onnx package (for example onnx-1.2.2 would have opset 7). By specifieing ```--opset``` the user can override the default to generate a graph with the desired opset. For example ```--opset 5``` would create a onnx graph that uses only ops available in opset 5. Because older opsets have in most cases fewer ops, some models might not convert on a older opset.
- custom-ops: the runtime may support custom ops that are not defined in onnx. A user can asked the converter to map to custom ops by listing them with the --custom-ops option. Tensorflow ops listed here will be mapped to a custom op of the same name as the tensorflow op but in the onnx domain ai.onnx.converters.tensorflow. For example: ```--custom-ops Print``` will insert a op ```Print``` in the onnx domain ```ai.onnx.converters.tensorflow``` into the graph. We also support a python api for custom ops documented later in this readme. 

Usage example (run following commands in tensorflow-onnx root directory):
```
python -m tf2onnx.convert\
    --input tests/models/fc-layers/frozen.pb\
    --inputs X:0\
    --outputs output:0\
    --output tests/models/fc-layers/model.onnx\
    --verbose
```
Some models specify placeholders with unknown ranks which can not be mapped to onnx. 
In those cases one can add the shape behind the input name in ```[]```, for example ```--inputs X:0[1,28,28,3]```

## <a name="summarize_graph"></a>Tool to get Graph Inputs & Outputs

To find the inputs and outputs for the TensorFlow graph the model developer will know or you can consult TensorFlow's [summarize_graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) tool, for example:
```
summarize_graph --in_graph=tests/models/fc-layers/frozen.pb
```
## <a name="freeze_graph"></a>Tool to Freeze Graph

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

# Testing
There are 2 types of tests.

## Unit test
```
python setup.py test
```

## Validate pre-trained TensorFlow models
```
python tests/run_pretrained_models.py
usage: run_pretrained_models.py [-h] [--cache CACHE] [--tests TESTS] [--backend BACKEND] [--verbose] [--debug] [--config yaml-config]

optional arguments:
  -h, --help         show this help message and exit
  --cache CACHE      pre-trained models cache dir
  --tests TESTS      tests to run
  --backend BACKEND  backend to use
  --config           yaml config file
  --verbose          verbose output
  --opset OPSET      target opset to use
  --perf csv-file    capture performance numbers or tensorflow and onnx runtime
  --debug            dump generated graph with shape info
```
```run_pretrained_models.py``` will run the TensorFlow model, captures the TensorFlow output and runs the same test against the specified ONNX backend after converting the model. The only practical backend to use at this time is Caffe2, and you need to install Caffe2 for this to work. 
If the option ```--perf csv-file``` is specified, we'll capture the eval runtime for tensorflow and onnx runtime and write the result into the given csv file.

You call it for example with:
```
python tests/run_pretrained_models.py --backend caffe2 --config tests/run_pretrained_models.yaml --perf perf.csv
```

# Using the Python Api
## Tensorflow to onnx conversion
In some cases it will be usefull to convert the models from tensorflow to onnx from a python script. You can use the following api:
```
import tf2onnx

tf2onnx.tfonnx.process_tf_graph(tf_graph, 
            continue_on_error=False, verbose=False, target=None,
            opset=None, custom_op_handlers=None,
            custom_rewriter=None, extra_opset=None):
    """Convert tensorflow graph to onnx graph.
        Args:
            tf_graph: tensorflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            verbose: print summary stats
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            custom_op_handlers: dictionary of custom ops handlers
            custom_rewriter: list of custom graph rewriters
        Return:
            onnx graph
    """
```
For example in [examples/call_coverter_via_python.py]():
```
import tensorflow as tf
import tf2onnx

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph)
    model_proto = onnx_graph.make_model("test", 
                                        ["input:0"], ["output:0"])
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
```
## Creating custom op mappings from python
For complex custom ops that require graph rewrites or input / attribute rewrites using the python interface to insert a custom op will be the eaiest way to accomplish the task.
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
    node.type = "Identity"
    node.domain = _TENSORFLOW_DOMAIN
    del node.input[1:]
    return node


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    x_ = tf.Print(x, [x], "hello")
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                 custom_op_handlers={"Print": print_handler},
                                                 extra_opset=[helper.make_opsetid(_TENSORFLOW_DOMAIN, 1)])
    model_proto = onnx_graph.make_model("test", ["input:0"], ["output:0"])
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
```

# How tf2onnx works
While the protobuf format of ONNX is not all that different than onnx, mileage will vary because TensorFlow supports 4x the ops compared to the current version of ONNX.
The converter needs to take care of a few things:
1. Convert the protobuf format. Since the format is similar this step is straight forward.
2. TensorFlow types need to be mapped to their ONNX equivalent.
3. For many ops TensorFlow passes parameters like shapes as inputs where ONNX wants to see them as attributes. Since we use a frozen graph, the converter will fetch the input as constant, converts it to an attribute and remove the original input.
4. TensorFlow in many cases composes ops out of multiple simpler ops. The converter will need to identify the subgraph for such ops, slice the subgraph out and replace it with the ONNX equivalent. This can become fairly complex so we use a graph matching library for it. A good example of this is the tensorflow transpose op.
5. TensorFlow's default data format is NHWC where ONNX requires NCHW. The converter will insert transpose ops to deal with this.
6. There are some ops like relu6 that are not supported in ONNX but the converter can be composed out of other ONNX ops.
7. ONNX backends are new and their implementations are not complete yet. For some ops the converter generate ops with deal with issues in existing backends.

### Step 1 - start with a frozen graph.
tf2onnx starts with a frozen graph. This is because of item 3 above.

### Step 2 - 1:1 convertion of the protobuf from tensorflow to onnx
tf2onnx first does a simple convertion from the TensorFlow protobuf format to the ONNX protobuf format without looking at individual ops.
We do this so we can use the ONNX graph as internal representation and write helper functions around it.
The code that does the conversion is in tensorflow_to_onnx(). tensorflow_to_onnx() will return the ONNX graph and a dictionary with shape information from TensorFlow. The shape information is helpful in some cases when processing individual ops. 
The ONNX graph is wrapped in a Graph object and nodes in the graph are wrapped in a Node object to allow easier graph manipulations on the graph. All code that deals with nodes and graphs is in graph.py.

### Step 3 - rewrite subgraphs
In the next step we apply graph matching code on the graph to re-write subgraphs for ops like transpose and lstm. For an example looks at rewrite_transpose().

### Step 4 - process individual ops
In the fourth step we look at individual ops that need attention. The dictionary _OPS_MAPPING will map tensorflow op types to a method that is used to process the op. The simplest case is direct_op() where the op can be taken as is. Whenever possible we try to group ops into common processing, for example all ops that require dealing with broadcasting are mapped to broadcast_op(). For an op that composes the tensorflow op from multiple onnx ops, see relu6_op().

### Step 5 - final processing
Once all ops are converted, we need to do a topological sort since ONNX requires it. process_tf_graph() is the method that takes care of all above steps.

# Extending tf2onnx
If you like to contribute and add new conversions to tf2onnx, the process is something like:
1. See if the op fits into one of the existing mappings. If so adding it to _OPS_MAPPING is all that is needed.
2. If the new op needs extra procesing, start a new mapping function.
3. If the tensorflow op is composed of multiple ops, consider using a graph re-write. While this might be a little harder initially, it works better for complex patterns.
4. Add a unit test in tests/test_backend.py. The unit tests mostly create the tensorflow graph, run it and capture the output, than convert to onnx, run against a onnx backend and compare tensorflow and onnx results. 
5. If there are pre-trained models that use the new op, consider adding those to test/run_pretrained_models.py.

# What is missing
- lstm/gru support (working on this)
- more testing
- more model coverage

# License

[MIT License](LICENSE)

