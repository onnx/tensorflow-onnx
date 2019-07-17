## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Const | ['1'] |
| ConstV2 | ['1'] |
| RandomNormal | ['1'] |
| RandomUniform | ['1'] |
| RandomNormalLike | ['1'] |
| RandomUniformLike | ['1'] |
| ZerosLike | ['1'] |
| LogicalNot | ['1'] |
| LogicalOr | ['1', '6'] |
| LogicalAnd | ['1', '6'] |
| Equal | ['1', '7'] |
| NotEqual | ['1', '7'] |
| Greater | ['1', '7'] |
| Less | ['1', '7'] |
| Add | ['1', '6'] |
| Div | ['1', '6'] |
| Mul | ['1', '6'] |
| Sub | ['1', '6'] |
| RealDiv | ['1', '6'] |
| TruncateDiv | ['1', '6'] |
| LeakyRelu | ['1'] |
| LogSoftmax | ['1'] |
| Softplus | ['1'] |
| Softsign | ['1'] |
| Abs | ['1', '6'] |
| Ceil | ['1', '6'] |
| Elu | ['1', '6'] |
| Exp | ['1', '6'] |
| Floor | ['1', '6'] |
| Log | ['1', '6'] |
| Neg | ['1', '6'] |
| Relu | ['1', '6'] |
| Sigmoid | ['1', '6'] |
| Sqrt | ['1', '6'] |
| Tanh | ['1', '6'] |
| Reciprocal | ['1', '6'] |
| Maximum | ['1'] |
| Minimum | ['1'] |
| Softmax | ['1'] |
| Square | ['1'] |
| Relu6 | ['1'] |
| Rsqrt | ['1'] |
| SquaredDifference | ['1'] |
| Sign | ['1', '9'] |
| Pow | ['1', '7'] |
| LRN | ['1'] |
| MatMul | ['1'] |
| BatchMatMul | ['1'] |
| Erf | ['1', '9'] |
| Selu | ['1'] |
| CheckNumerics | ['1'] |
| StopGradient | ['1'] |
| Placeholder | ['1'] |
| PlaceholderV2 | ['1'] |
| PlaceholderWithDefault | ['1'] |
| NoOp | ['1'] |
| Size | ['1'] |
| Flatten | ['1', '9'] |
| Dropout | ['1', '6', '7', '10'] |
| Identity | ['1'] |
| Reshape | ['1', '5'] |
| Squeeze | ['1'] |
| Transpose | ['1'] |
| Concat | ['1'] |
| ConcatV2 | ['1'] |
| Slice | ['1', '10'] |
| Gather | ['1'] |
| GatherV2 | ['1'] |
| GatherNd | ['1'] |
| Split | ['1', '2'] |
| SplitV | ['1', '2'] |
| ExpandDims | ['1', '7'] |
| StridedSlice | ['1', '10'] |
| Cast | ['1', '6', '9'] |
| TopKV2 | ['1', '10'] |
| Tile | ['1'] |
| Pack | ['1'] |
| Unpack | ['1'] |
| OneHot | ['1', '9'] |
| Shape | ['1'] |
| BatchToSpaceND | ['1'] |
| SpaceToBatchND | ['1'] |
| Conv1D | ['1'] |
| Conv2D | ['1'] |
| Conv3D | ['1'] |
| Conv2DBackpropInput | ['1'] |
| DepthwiseConv2d | ['1'] |
| DepthwiseConv2dNative | ['1'] |
| MaxPool | ['1', '10'] |
| MaxPoolV2 | ['1', '10'] |
| AvgPool | ['1', '10'] |
| AvgPool3D | ['1', '10'] |
| BiasAdd | ['1', '7'] |
| BiasAddV1 | ['1', '7'] |
| Pad | ['1'] |
| PadV2 | ['1'] |
| MirrorPad | ['1'] |
| SpaceToDepth | ['1'] |
| DepthToSpace | ['1'] |
| Prod | ['1'] |
| Sum | ['1'] |
| Mean | ['1'] |
| Max | ['1'] |
| Min | ['1'] |
| ArgMax | ['1'] |
| ArgMin | ['1'] |
| LSTMBlockCell | ['1', '7'] |
| FloorDiv | ['6'] |
| FusedBatchNorm | ['6', '9'] |
| FusedBatchNormV2 | ['6', '9'] |
| All | ['6'] |
| Any | ['6'] |
| AddN | ['6'] |
| If | ['7'] |
| Loop | ['7'] |
| Scan | ['7'] |
| Range | ['7'] |
| Fill | ['7', '9'] |
| Multinomial | ['7'] |
| LessEqual | ['7'] |
| GreaterEqual | ['7'] |
| Acos | ['7'] |
| Asin | ['7'] |
| Atan | ['7'] |
| Cos | ['7'] |
| Sin | ['7'] |
| Tan | ['7'] |
| FloorMod | ['7'] |
| ResizeBilinear | ['7', '9', '10'] |
| ResizeNearestNeighbor | ['7', '9', '10'] |
| MatrixBandPart | ['7'] |
| SoftmaxCrossEntropyWithLogits | ['7'] |
| SparseSoftmaxCrossEntropyWithLogits | ['7', '9'] |
| Select | ['8', '9'] |
| ReverseSequence | ['8', '9', '10'] |
| MaxPoolWithArgmax | ['8'] |
| Where | ['9'] |
| Acosh | ['9'] |
| Asinh | ['9'] |
| Atanh | ['9'] |
| Cosh | ['9'] |
| Sinh | ['9'] |
| IsNan | ['9'] |
| IsInf | ['10'] |
| NonMaxSuppressionV2 | ['10'] |
| NonMaxSuppressionV3 | ['10'] |
### Domain: "com.microsoft" 
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Range | ['1'] |
