/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


// opregister
REGISTER_OP("AddOne")
  .Input("add_one: int32")
  .Output("oneed: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


// keneldefinition
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
    for (int i = 0; i < N; i++) {
      output(i) += 1;
    }
#endif
    if (N > 0) {
      output(0) = input(0);
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);