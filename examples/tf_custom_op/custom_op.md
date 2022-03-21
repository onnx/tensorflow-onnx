<!--- SPDX-License-Identifier: Apache-2.0 -->

## Example of converting TensorFlow model with custom op to ONNX

This document describes the ways for exporting TensorFlow model with a custom operator, exporting the operator to ONNX format, and adding the operator to ONNX Runtime for model inference. Tensorflow provides abundant set of operators, and also provides the extending implmentation to register as the new operators. The new custom operators are usually not recognized by tf2onnx conversion and onnxruntime. So the TensorFlow custom ops should be exported using a combination of existing and/or new custom ONNX ops. Once the operator is converted to ONNX format, users can implement and register it with ONNX Runtime for model inference. This document explains the details of this process end-to-end, along with an example.


### Required Steps

  - [1](#step1) - Adding the Tensorflow custom operator implementation in C++ and registering it with TensorFlow
  - [2](#step2) - Exporting the custom Operator to ONNX, using:
  <br />             - a combination of existing ONNX ops
  <br />              or
  <br />              - a custom ONNX Operator
  - [3](#step3) - Adding the custom operator implementation and registering it in ONNX Runtime (required only if using a custom ONNX op in step 2)


### Implement the Custom Operator
Firstly, try to install the TensorFlow latest version (Nighly is better) build refer to [here](https://github.com/tensorflow/tensorflow#install). And then implement the custom operators saving in TensorFlow library format and the file usually ends with `.so`. We have a simple example of `AddOne`, which is adding one for a tensor.


#### Define the op interface
Specify the name of your op, its inputs (types and names) and outputs (types and names), as well as docstrings and any attrs the op might require.
```
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("AddOne")
  .Input("add_one: int32")
  .Output("result: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });
```

#### Implement the op kernel
Create a class that extends `OpKernel` and overrides the `Compute()` method. The implementation is written to the function `Compute()`.

```
#include "tensorflow/core/framework/op_kernel.h"

void AddOneKernelLauncher(const Tensor* t_in, const int n, Tensor* t_out);

class AddOneOp : public OpKernel {
public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Tensore in input
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Tensore in output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<int32>();

#if GOOGLE_CUDA
    AddOneKernelLauncher(input, input.size(), output);
#else
    const int N = input.size();
    for (int i = 0; i < N; i++) output(i) += 1;
#endif
    if (N > 0) output(0) = input(0);
  }
};
```
Add the Register kernel build,
```
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);
```
Save below code in C++ `.cc` file,

#### Using C++ compiler to compile the op
Assuming you have g++ installed, here is the sequence of commands you can use to compile your op into a dynamic library.
```
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared add_one.cc -o add_one.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```
After below steps, we can get a TensorFlow custom op library `add_one.so`.


### Convert the Operator to ONNX
To be able to use this custom ONNX operator for inference, we need to add our custom operator to an inference engine. If the operator can be conbinded with exsiting [ONNX standard operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md). The case will be easier: 

1- using [--load_op_libraries](https://github.com/onnx/tensorflow-onnx#--load_op_libraries) in conversion command or `tf.load_op_library()` method in code to load the TensorFlow custom ops library.

2- implement the op handler, registered it with the `@tf_op` decorator. Those handlers will be registered via the decorator on load of the module. [Here](https://github.com/onnx/tensorflow-onnx/tree/main/tf2onnx/onnx_opset) are examples of TensorFlow op hander implementations.

```
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import os
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.tf_loader import tf_placeholder


DIR_PATH = os.path.realpath(os.path.dirname(__file__))
saved_model_path = os.path.join(DIR_PATH, "model.onnx")
tf_library_path = os.path.join(DIR_PATH, "add_one.so")


@tf_op("AddOne", onnx_op="Add")
class AddOne:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node_shape = ctx.get_shape(node.input[0])
        const_one = ctx.make_const(utils.make_name("const_one"), np.ones(node_shape, dtype = np.int32)).output[0]
        node.input.append(const_one)

@tf.function
def func(x):
    AddOne = tf.load_op_library(tf_library_path)
    x_ = AddOne.add_one(x)
    output = tf.identity(x_, name="output")
    return output

spec = [tf.TensorSpec(shape=(2, 3), dtype=tf.int32, name="input")]

onnx_model, _ = tf2onnx.convert.from_function(func, input_signature=spec, opset=15)

with open(saved_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

onnx_model = onnx.load(saved_model_path)
onnx.checker.check_model(onnx_model)
```

3- Run in ONNXRuntime, using `InferenceSession` to do inference and get the result.
```
import onnxruntime as ort
input = np.arange(6).reshape(2,3).astype(np.int32)
ort_session = ort.InferenceSession(saved_model_path)
ort_inputs = {ort_session.get_inputs()[0].name: input}

ort_outs = ort_session.run(None, ort_inputs)
print("input:", input, "\nAddOne ort_outs:", ort_outs)
```


If the operator can not using existing ONNX standard operators only, you need to go to [implement the operator in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md#implement-the-operator-in-onnx-runtime).

### References:
1- [Create an custom TensorFlow op](https://www.tensorflow.org/guide/create_op)

2- [ONNX Runtime: Adding a New Op](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html#register-a-custom-operator)

3- [PyTorch Custom Operators export to ONNX](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)