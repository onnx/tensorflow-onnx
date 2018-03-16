tf2onnx - convert tensorflow models to onnx models.
========

Tf2onnx converts a tensorflow graph to an onnx graph.

Tf2onnx is in its early development. Mileage will vary since tensorflow supports ~4 times the operations that the current onnx version supports. But standard models seem to be using mostly ops that onnx does support.

# Status
Baisc net and conv nets should work. A list of models that pass tests can be found [here](tests/run_pretrained_models.yaml)

# Installation
Install dependencies:
```
pip install onnx
pip install tensorflow
```
If you want to run unit tests against the caffe2 onnx backend, build and install caffe2 and onnx-caffe2:
```
https://github.com/caffe2/caffe2
https://github.com/onnx/onnx-caffe2
```
Once dependencies are installed, from the tf2onnx root folder call:
```
python setup.py install
```
or 
```
python setup.py develop
```

To create a wheel for distribution:
```
python setup.py bdist_wheel
```
# Usage
```
python -m tf2onnx.convert
usage: convert.py [-h] --input INPUT [--output OUTPUT] --inputs INPUTS --outputs OUTPUTS [--pbtxt PBTXT] [--pretty] [--continue_on_error] [--verbose]
```
For example:
```
python -m tf2onnx.convert.py --input tests/models/fc-layers/frozen.pb --inputs X:0 --outputs output:0 --output tests/models/fc-layers/model.onnx --pretty --verbose
```

To convert a tensorflow model, tf2onnx expects a ```frozen tensorflow graph``` and the user needs to specify inputs and outputs for the graph. 
To find the inputs and outputs for the tensorflow graph the model developer will know or you can consult tensorflow's [summarize_graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) tool, for example:
```
summarize_graph --in_graph=tests/models/fc-layers/frozen.pb
```

The tensorflow tool to freeze the graph is [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

For example:
```
tools=`python -c "import tensorflow as tf; print(tf.sysconfig.get_lib()+'/python/tools')"`

python $tools/freeze_graph.py \
    --input_graph=my_checkpoint_dir/graphdef.pb \
    --input_binary=true \
    --input_names=input:0 \
    --output_node_names=output:0 \
    --input_checkpoint=my_checkpoint_dir \
    --output_graph=tests/models/fc-layers/frozen.pb
```


# Testing
There are 2 types of tests.

## Unit test
```
python setup.py test
```

## Validate pre-trained tensorflow models
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
  --debug            dump generated graph with shape info
```
```run_pretrained_models.py``` will run the tensorflow model, captures the tensorflow output and runs the same test against the specified onnx backend after converting the model. The only practical backend to use at this time is caffe2, and you need to install caffe2 for this to work.

You call it for example with:
```
python tests/run_pretrained_models.py --backend caffe2 --config tests/run_pretrained_models.yaml
```
# How tf2onnx works
While the protobuf format of onnx is not all that different than onnx, mileage will vary because tensorflow supports 4x the ops compared to the current version of onnx.
The converter needs to take care of a few things:
1. Convert the protobuf format. Since the format is similar this step is straight forward.
2. Tensorflow types need to be mapped to their onnx equivalent.
3. For many ops tensorflow passes parameters like shapes as inputs where onnx wants to see them as attributes. Since we use a frozen graph, the converter will fetch the input as constant, converts it to an attribute and remove the original input.
4. Tensorflow in many cases composes ops out of multiple simpler ops. The converter will need to identify the subgraph for such ops, slice the subgraph out and replace it with the onnx equivalent. This can become fairly complex so we use a graph matching library for it. A good example of this is the tensorflow transpose op.
5. Tensorflow's default data format is NHWC where onnx requires NCHW. The converter will insert transpose ops to deal with this.
6. There are some ops like relu6 that are not supported in onnx but the converter can be composed out of other onnx ops.
7. Onnx backends are new and their implementations are not complete yet. For some ops the converter generate ops with deal with issues in existing backends.

### Step 1 - start with a frozen graph.
tf2onnx starts with a frozen graph. This is because of item 3 above.

### Step 2 - 1:1 convertion of the protobuf from tensorflow to onnx
tf2onnx first does a simple convertion from the tensorflow protobuf format to the onnx protobuf format without looking at individual ops.
We do this so we can use the onnx graph as internal representation and write helper functions around it.
The code that does the conversion is in tensorflow_to_onnx(). tensorflow_to_onnx() will return the onnx graph and a dictionary with shape information from tensorflow. The shape information is helpfull in some cases when processing individual ops. 
The onnx graph is wrapped in a Graph object and nodes in the graph are wrapped in a Node object to allow easier graph manipulations on the graph. All code that deals with nodes and graphs is in graph.py.

### Step 3 - rewrite subgraphs
In the next step we apply graph matching code on the graph to re-write subgraphs for ops like transpose and lstm. For an example looks at rewrite_transpose().

### Step 4 - process individual ops
In the fourth step we look at individual ops that need attention. The dictionary _OPS_MAPPING will map tensorflow op types to a method that is used to process the op. The simplest case is direct_op() where the op can be taken as is. Whenever possible we try to group ops into common processing, for example all ops that require dealing with broadcasting are mapped to broadcast_op(). For an op that composes the tensorflow op from multiple onnx ops, see relu6_op().

### Step 5 - final processing
Once all ops are converted, we need to do a topological sort since onnx requires it. process_tf_graph() is the method that takes care of all above steps.

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
- mode model coverage

# License

[MIT License](LICENSE)

