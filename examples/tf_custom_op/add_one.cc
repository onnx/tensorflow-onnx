/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


// opregister
REGISTER_OP("AddOne")
  .Input("add_one: int32")
  .Output("result: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


// keneldefinition
#include "tensorflow/core/framework/op_kernel.h"

class AddOneOp : public OpKernel {
public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Tensor in input
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Tensor in output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<int32>();

    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output(i) += 1;
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);
