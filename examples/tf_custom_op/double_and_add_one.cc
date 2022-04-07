/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;


// opregister
REGISTER_OP("DoubleAndAddOne")
  .Input("x: T")
  .Output("result: T")
  .Attr("T: {float, double, int32}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    ::tensorflow::shape_inference::ShapeHandle shape_x = c->input(0);
    if (!c->RankKnown(shape_x)) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    }
    c->set_output(0, shape_x);
    return Status::OK();
  })
  .Doc(R"doc(
    Calculate the value 2x + 1. 
    x: A Tensor `Tensor`. Must be one of the types in `T`.

    Returns: A `Tensor`. Has the same type with `x`.
  )doc");


// keneldefinition
template <typename T>
class DoubleAndAddOneOp : public OpKernel {
public:
  explicit DoubleAndAddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Tensor in output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<T>();

    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output(i) = output(i) * T(2) + T(1);
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("DoubleAndAddOne")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        DoubleAndAddOneOp<float>);
REGISTER_KERNEL_BUILDER(Name("DoubleAndAddOne")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        DoubleAndAddOneOp<double>);
REGISTER_KERNEL_BUILDER(Name("DoubleAndAddOne")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int>("T"),
                        DoubleAndAddOneOp<int>);


#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("DoubleAndAddOne").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DoubleAndAddOneOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

