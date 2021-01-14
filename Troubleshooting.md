tf2onnx - common issues when converting models.
========

## tensorflow op is not supported
Example:

```ValueError: tensorflow op NonMaxSuppression is not supported``` 

means that the given tensorflow op is not mapped to ONNX. This could have multiple reasons:

(1) we have not gotten to implement it. NonMaxSuppression is such an example: we implemented NonMaxSuppressionV2 and NonMaxSuppressionV3 but not the older NonMaxSuppression op.

To get this fixed you can open an issue or send us a PR with a fix.

(2) There is no direct mapping to ONNX.

Sometimes there is no direct mapping from tensorflow to ONNX. We took care are of the most common cases. But for less frequently used ops there might be a mapping missing. To get this fixed there 2 options:

a) in tf2onnx you can compose the op out of different ops. A good example for this the [Erf op](https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/onnx_opset/math.py#L317). Before opset-9 this tf2onnx composes Erf with other ONNX ops.

b) You request the missing op to be added to [ONNX](https://github.com/onnx/onnx). After it is added to ONNX and some runtime implements it we'll add it to tf2onnx. You can see that this happened for the Erf Op. Starting with opset-9, ONNX added it - tf2onnx no longer composes the op and instead passes it to ONNX.

c) The op is too complex to compose and it's to exotic to add to ONNX. In that cases you can use a custom op to implement it. Custom ops are documented in the [README](README.md) and there is an example [here](https://github.com/onnx/tensorflow-onnx/blob/master/examples/custom_op_via_python.py). There are 2 flavors of it:
- you could compose the functionality by using multiple ONNX ops.
- you can implement the op in your runtime as custom op (assuming that most runtimes do have such a mechanism) and then map it in tf2onnx as custom op.

## get tensor value: ... must be Const

There is a common group of errors that reports ```get tensor value: ... must be Const```.
The reason for this is that there is a dynamic input of a tensorflow op but the equivalent ONNX op uses a static attribute. In other words in tensorflow that input is only known at runtime but in ONNX it need to be known at graph creation time.

An example of this is the [ONNX Slice operator before opset-10](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-1) - the start and end of the slice are static attributes that need to be known at graph creation. In tensorflow the [strided slice op](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/strided-slice) allows dynamic inputs. tf2onnx will try to find the real value of begin and end of the slice and can find them in most cases. But if those are real dynamic values calculate at runtime it will result in the message ```get tensor value: ... must be Const```.

You can pass the options ```--fold_const``` in the tf2onnx command line that allows tf2onnx to apply more aggressive constant folding which will increase chances to find a constant.

If this doesn't work the model is most likely not to be able to convert to ONNX. We used to see this a lot of issue with the ONNX Slice op and in opset-10 was updated for exactly this reason.
