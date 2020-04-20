## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 11 |
| Acos | 7 ~ 11 |
| Acosh | 9 ~ 11 |
| Add | 1 ~ 11 |
| AddN | 6 ~ 11 |
| AddV2 | 1 ~ 11 |
| All | 6 ~ 11 |
| Any | 6 ~ 11 |
| ArgMax | 1 ~ 11 |
| ArgMin | 1 ~ 11 |
| Asin | 7 ~ 11 |
| Asinh | 9 ~ 11 |
| Atan | 7 ~ 11 |
| Atanh | 9 ~ 11 |
| AvgPool | 1 ~ 11 |
| AvgPool3D | 1 ~ 11 |
| BatchMatMul | 1 ~ 11 |
| BatchMatMulV2 | 1 ~ 11 |
| BatchToSpaceND | 1 ~ 11 |
| BiasAdd | 1 ~ 11 |
| BiasAddV1 | 1 ~ 11 |
| BroadcastTo | 8 ~ 11 |
| Cast | 1 ~ 11 |
| Ceil | 1 ~ 11 |
| CheckNumerics | 1 ~ 11 |
| ClipByValue | 8 ~ 11 |
| Concat | 1 ~ 11 |
| ConcatV2 | 1 ~ 11 |
| Const | 1 ~ 11 |
| ConstV2 | 1 ~ 11 |
| Conv1D | 1 ~ 11 |
| Conv2D | 1 ~ 11 |
| Conv2DBackpropInput | 1 ~ 11 |
| Conv3D | 1 ~ 11 |
| Cos | 7 ~ 11 |
| Cosh | 9 ~ 11 |
| CropAndResize | 10 ~ 11 |
| CudnnRNN | 10 ~ 11 |
| Cumsum | 11 |
| DepthToSpace | 1 ~ 11 |
| DepthwiseConv2d | 1 ~ 11 |
| DepthwiseConv2dNative | 1 ~ 11 |
| Div | 1 ~ 11 |
| Dropout | 1 ~ 11 |
| Elu | 1 ~ 11 |
| Equal | 1 ~ 11 |
| Erf | 1 ~ 11 |
| Exp | 1 ~ 11 |
| ExpandDims | 1 ~ 11 |
| FIFOQueueV2 | 8 ~ 11 |
| Fill | 7 ~ 11 |
| Flatten | 1 ~ 11 |
| Floor | 1 ~ 11 |
| FloorDiv | 6 ~ 11 |
| FloorMod | 7 ~ 11 |
| FusedBatchNorm | 6 ~ 11 |
| FusedBatchNormV2 | 6 ~ 11 |
| FusedBatchNormV3 | 6 ~ 11 |
| Gather | 1 ~ 11 |
| GatherNd | 1 ~ 11 |
| GatherV2 | 1 ~ 11 |
| Greater | 1 ~ 11 |
| GreaterEqual | 7 ~ 11 |
| HashTableV2 | 8 ~ 11 |
| Identity | 1 ~ 11 |
| IdentityN | 1 ~ 11 |
| If | 1 ~ 11 |
| IsInf | 10 ~ 11 |
| IsNan | 9 ~ 11 |
| IteratorGetNext | 8 ~ 11 |
| IteratorV2 | 8 ~ 11 |
| LRN | 1 ~ 11 |
| LSTMBlockCell | 1 ~ 11 |
| LeakyRelu | 1 ~ 11 |
| LeftShift | 11 |
| Less | 1 ~ 11 |
| LessEqual | 7 ~ 11 |
| Log | 1 ~ 11 |
| LogSoftmax | 1 ~ 11 |
| LogicalAnd | 1 ~ 11 |
| LogicalNot | 1 ~ 11 |
| LogicalOr | 1 ~ 11 |
| LookupTableFindV2 | 8 ~ 11 |
| Loop | 7 ~ 11 |
| MatMul | 1 ~ 11 |
| MatrixBandPart | 7 ~ 11 |
| MatrixDeterminant | 11 |
| MatrixDiagPart | 11 |
| Max | 1 ~ 11 |
| MaxPool | 1 ~ 11 |
| MaxPoolV2 | 1 ~ 11 |
| MaxPoolWithArgmax | 8 ~ 11 |
| Maximum | 1 ~ 11 |
| Mean | 1 ~ 11 |
| Min | 1 ~ 11 |
| Minimum | 1 ~ 11 |
| MirrorPad | 1 ~ 11 |
| Mul | 1 ~ 11 |
| Multinomial | 7 ~ 11 |
| Neg | 1 ~ 11 |
| NoOp | 1 ~ 11 |
| NonMaxSuppressionV2 | 10 ~ 11 |
| NonMaxSuppressionV3 | 10 ~ 11 |
| NonMaxSuppressionV4 | 10 ~ 11 |
| NonMaxSuppressionV5 | 10 ~ 11 |
| NotEqual | 1 ~ 11 |
| OneHot | 1 ~ 11 |
| Pack | 1 ~ 11 |
| Pad | 1 ~ 11 |
| PadV2 | 1 ~ 11 |
| Placeholder | 1 ~ 11 |
| PlaceholderV2 | 1 ~ 11 |
| PlaceholderWithDefault | 1 ~ 11 |
| Pow | 1 ~ 11 |
| Prod | 1 ~ 11 |
| QueueDequeueV2 | 8 ~ 11 |
| RandomNormal | 1 ~ 11 |
| RandomNormalLike | 1 ~ 11 |
| RandomUniform | 1 ~ 11 |
| RandomUniformLike | 1 ~ 11 |
| Range | 7 ~ 11 |
| RealDiv | 1 ~ 11 |
| Reciprocal | 1 ~ 11 |
| Relu | 1 ~ 11 |
| Relu6 | 1 ~ 11 |
| Reshape | 1 ~ 11 |
| ResizeBilinear | 7 ~ 11 |
| ResizeNearestNeighbor | 7 ~ 11 |
| ReverseSequence | 8 ~ 11 (Except 9) |
| ReverseV2 | 10 ~ 11 |
| RightShift | 11 |
| Round | 11 |
| Rsqrt | 1 ~ 11 |
| Scan | 7 ~ 11 |
| ScatterNd | 11 |
| Select | 7 ~ 11 |
| SelectV2 | 7 ~ 11 |
| Selu | 1 ~ 11 |
| Shape | 1 ~ 11 |
| Sigmoid | 1 ~ 11 |
| Sign | 1 ~ 11 |
| Sin | 7 ~ 11 |
| Sinh | 9 ~ 11 |
| Size | 1 ~ 11 |
| Slice | 1 ~ 11 |
| Softmax | 1 ~ 11 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 11 |
| Softplus | 1 ~ 11 |
| Softsign | 1 ~ 11 |
| SpaceToBatchND | 1 ~ 11 |
| SpaceToDepth | 1 ~ 11 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 11 |
| Split | 1 ~ 11 |
| SplitV | 1 ~ 11 |
| Sqrt | 1 ~ 11 |
| Square | 1 ~ 11 |
| SquaredDifference | 1 ~ 11 |
| Squeeze | 1 ~ 11 |
| StatelessIf | 1 ~ 11 |
| StatelessWhile | 7 ~ 11 |
| StopGradient | 1 ~ 11 |
| StridedSlice | 1 ~ 11 |
| Sub | 1 ~ 11 |
| Sum | 1 ~ 11 |
| Tan | 7 ~ 11 |
| Tanh | 1 ~ 11 |
| TensorListFromTensor | 7 ~ 11 |
| TensorListGetItem | 7 ~ 11 |
| TensorListLength | 7 ~ 11 |
| TensorListReserve | 7 ~ 11 |
| TensorListResize | 7 ~ 11 |
| TensorListSetItem | 7 ~ 11 |
| TensorListStack | 7 ~ 11 |
| Tile | 1 ~ 11 |
| TopKV2 | 1 ~ 11 |
| Transpose | 1 ~ 11 |
| TruncateDiv | 1 ~ 11 |
| Unique | 11 |
| Unpack | 1 ~ 11 |
| Where | 9 ~ 11 |
| While | 7 ~ 11 |
| ZerosLike | 1 ~ 11 |
### Domain: "com.microsoft" 
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Conv2DBackpropInput | 1 |
| CropAndResize | 1 |
| Range | 1 |
