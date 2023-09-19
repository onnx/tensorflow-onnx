<!--- SPDX-License-Identifier: Apache-2.0 -->

## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 18 |
| Acos | 7 ~ 18 |
| Acosh | 9 ~ 18 |
| Add | 1 ~ 18 |
| AddN | 6 ~ 18 |
| AddV2 | 1 ~ 18 |
| AdjustContrastv2 | 1 ~ 18 |
| AdjustHue | 11 ~ 18 |
| AdjustSaturation | 11 ~ 18 |
| All | 6 ~ 18 |
| Any | 6 ~ 18 |
| ArgMax | 1 ~ 18 |
| ArgMin | 1 ~ 18 |
| AsString | 9 ~ 18 |
| Asin | 7 ~ 18 |
| Asinh | 9 ~ 18 |
| Atan | 7 ~ 18 |
| Atan2 | 9 ~ 18 |
| Atanh | 9 ~ 18 |
| AvgPool | 1 ~ 18 |
| AvgPool3D | 1 ~ 18 |
| BatchMatMul | 1 ~ 18 |
| BatchMatMulV2 | 1 ~ 18 |
| BatchMatMulV3 | 1 ~ 18 |
| BatchToSpaceND | 1 ~ 18 |
| BiasAdd | 1 ~ 18 |
| BiasAddV1 | 1 ~ 18 |
| Bincount | 11 ~ 18 |
| BitwiseAnd | 18 |
| BitwiseOr | 18 |
| BitwiseXor | 18 |
| BroadcastTo | 8 ~ 18 |
| CTCGreedyDecoder | 11 ~ 18 |
| Cast | 1 ~ 18 |
| Ceil | 1 ~ 18 |
| CheckNumerics | 1 ~ 18 |
| ClipByValue | 8 ~ 18 |
| CombinedNonMaxSuppression | 12 ~ 18 |
| ComplexAbs | 1 ~ 18 |
| Concat | 1 ~ 18 |
| ConcatV2 | 1 ~ 18 |
| Const | 1 ~ 18 |
| ConstV2 | 1 ~ 18 |
| Conv1D | 1 ~ 18 |
| Conv2D | 1 ~ 18 |
| Conv2DBackpropInput | 1 ~ 18 |
| Conv3D | 1 ~ 18 |
| Conv3DBackpropInputV2 | 1 ~ 18 |
| Cos | 7 ~ 18 |
| Cosh | 9 ~ 18 |
| CropAndResize | 10 ~ 18 |
| CudnnRNN | 10 ~ 18 |
| Cumsum | 11 ~ 18 |
| DenseBincount | 11 ~ 18 |
| DenseToDenseSetOperation | 11 ~ 18 |
| DepthToSpace | 1 ~ 18 |
| DepthwiseConv2d | 1 ~ 18 |
| DepthwiseConv2dNative | 1 ~ 18 |
| Div | 1 ~ 18 |
| DivNoNan | 9 ~ 18 |
| Dropout | 1 ~ 18 |
| DynamicPartition | 9 ~ 18 |
| DynamicStitch | 10 ~ 18 |
| Einsum | 12 ~ 18 |
| Elu | 1 ~ 18 |
| EnsureShape | 1 ~ 18 |
| Equal | 1 ~ 18 |
| Erf | 1 ~ 18 |
| Exp | 1 ~ 18 |
| ExpandDims | 1 ~ 18 |
| FFT | 1 ~ 18 |
| FIFOQueueV2 | 8 ~ 18 |
| FakeQuantWithMinMaxArgs | 10 ~ 18 |
| FakeQuantWithMinMaxVars | 10 ~ 18 |
| Fill | 7 ~ 18 |
| Flatten | 1 ~ 18 |
| Floor | 1 ~ 18 |
| FloorDiv | 6 ~ 18 |
| FloorMod | 7 ~ 18 |
| FusedBatchNorm | 6 ~ 18 |
| FusedBatchNormV2 | 6 ~ 18 |
| FusedBatchNormV3 | 6 ~ 18 |
| Gather | 1 ~ 18 |
| GatherNd | 1 ~ 18 |
| GatherV2 | 1 ~ 18 |
| Greater | 1 ~ 18 |
| GreaterEqual | 7 ~ 18 |
| HardSwish | 14 ~ 18 |
| HashTableV2 | 8 ~ 18 |
| Identity | 1 ~ 18 |
| IdentityN | 1 ~ 18 |
| If | 1 ~ 18 |
| Invert | 18 |
| InvertPermutation | 11 ~ 18 |
| IsFinite | 10 ~ 18 |
| IsInf | 10 ~ 18 |
| IsNan | 9 ~ 18 |
| IteratorGetNext | 8 ~ 18 |
| IteratorV2 | 8 ~ 18 |
| LRN | 1 ~ 18 |
| LSTMBlockCell | 1 ~ 18 |
| LeakyRelu | 1 ~ 18 |
| LeftShift | 11 ~ 18 |
| Less | 1 ~ 18 |
| LessEqual | 7 ~ 18 |
| Log | 1 ~ 18 |
| LogSoftmax | 1 ~ 18 |
| LogicalAnd | 1 ~ 18 |
| LogicalNot | 1 ~ 18 |
| LogicalOr | 1 ~ 18 |
| LookupTableFindV2 | 8 ~ 18 |
| LookupTableSizeV2 | 1 ~ 18 |
| Loop | 7 ~ 18 |
| MatMul | 1 ~ 18 |
| MatrixBandPart | 7 ~ 18 |
| MatrixDeterminant | 11 ~ 18 |
| MatrixDiag | 12 ~ 18 |
| MatrixDiagPart | 11 ~ 18 |
| MatrixDiagPartV2 | 11 ~ 18 |
| MatrixDiagPartV3 | 11 ~ 18 |
| MatrixDiagV2 | 12 ~ 18 |
| MatrixDiagV3 | 12 ~ 18 |
| MatrixSetDiagV3 | 12 ~ 18 |
| Max | 1 ~ 18 |
| MaxPool | 1 ~ 18 |
| MaxPool3D | 1 ~ 18 |
| MaxPoolV2 | 1 ~ 18 |
| MaxPoolWithArgmax | 8 ~ 18 |
| Maximum | 1 ~ 18 |
| Mean | 1 ~ 18 |
| Min | 1 ~ 18 |
| Minimum | 1 ~ 18 |
| MirrorPad | 1 ~ 18 |
| Mul | 1 ~ 18 |
| Multinomial | 7 ~ 18 |
| Neg | 1 ~ 18 |
| NoOp | 1 ~ 18 |
| NonMaxSuppressionV2 | 10 ~ 18 |
| NonMaxSuppressionV3 | 10 ~ 18 |
| NonMaxSuppressionV4 | 10 ~ 18 |
| NonMaxSuppressionV5 | 10 ~ 18 |
| NotEqual | 1 ~ 18 |
| OneHot | 1 ~ 18 |
| Pack | 1 ~ 18 |
| Pad | 1 ~ 18 |
| PadV2 | 1 ~ 18 |
| ParallelDynamicStitch | 10 ~ 18 |
| Placeholder | 1 ~ 18 |
| PlaceholderV2 | 1 ~ 18 |
| PlaceholderWithDefault | 1 ~ 18 |
| Pow | 1 ~ 18 |
| Prelu | 1 ~ 18 |
| Prod | 1 ~ 18 |
| QueueDequeueManyV2 | 8 ~ 18 |
| QueueDequeueUpToV2 | 8 ~ 18 |
| QueueDequeueV2 | 8 ~ 18 |
| RFFT | 1 ~ 18 |
| RFFT2D | 1 ~ 18 |
| RaggedGather | 11 ~ 18 |
| RaggedRange | 11 ~ 18 |
| RaggedTensorFromVariant | 13 ~ 18 |
| RaggedTensorToSparse | 11 ~ 18 |
| RaggedTensorToTensor | 11 ~ 18 |
| RaggedTensorToVariant | 13 ~ 18 |
| RandomNormal | 1 ~ 18 |
| RandomNormalLike | 1 ~ 18 |
| RandomShuffle | 10 ~ 18 |
| RandomStandardNormal | 1 ~ 18 |
| RandomUniform | 1 ~ 18 |
| RandomUniformInt | 1 ~ 18 |
| RandomUniformLike | 1 ~ 18 |
| Range | 7 ~ 18 |
| RealDiv | 1 ~ 18 |
| Reciprocal | 1 ~ 18 |
| Relu | 1 ~ 18 |
| Relu6 | 1 ~ 18 |
| Reshape | 1 ~ 18 |
| ResizeArea | 7 ~ 18 |
| ResizeBicubic | 7 ~ 18 |
| ResizeBilinear | 7 ~ 18 |
| ResizeNearestNeighbor | 7 ~ 18 |
| ReverseSequence | 8 ~ 18 (Except 9) |
| ReverseV2 | 10 ~ 18 |
| RightShift | 11 ~ 18 |
| Rint | 11 ~ 18 |
| Roll | 10 ~ 18 |
| Round | 1 ~ 18 |
| Rsqrt | 1 ~ 18 |
| SampleDistortedBoundingBox | 9 ~ 18 |
| SampleDistortedBoundingBoxV2 | 9 ~ 18 |
| Scan | 7 ~ 18 |
| ScatterNd | 11 ~ 18 |
| SegmentMax | 11 ~ 18 |
| SegmentMean | 11 ~ 18 |
| SegmentMin | 11 ~ 18 |
| SegmentProd | 11 ~ 18 |
| SegmentSum | 11 ~ 18 |
| Select | 7 ~ 18 |
| SelectV2 | 7 ~ 18 |
| Selu | 1 ~ 18 |
| Shape | 1 ~ 18 |
| Sigmoid | 1 ~ 18 |
| Sign | 1 ~ 18 |
| Sin | 7 ~ 18 |
| Sinh | 9 ~ 18 |
| Size | 1 ~ 18 |
| Slice | 1 ~ 18 |
| Softmax | 1 ~ 18 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 18 |
| Softplus | 1 ~ 18 |
| Softsign | 1 ~ 18 |
| SpaceToBatchND | 1 ~ 18 |
| SpaceToDepth | 1 ~ 18 |
| SparseFillEmptyRows | 11 ~ 18 |
| SparseReshape | 11 ~ 18 |
| SparseSegmentMean | 11 ~ 18 |
| SparseSegmentMeanWithNumSegments | 11 ~ 18 |
| SparseSegmentSqrtN | 11 ~ 18 |
| SparseSegmentSqrtNWithNumSegments | 11 ~ 18 |
| SparseSegmentSum | 11 ~ 18 |
| SparseSegmentSumWithNumSegments | 11 ~ 18 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 18 |
| SparseToDense | 11 ~ 18 |
| Split | 1 ~ 18 |
| SplitV | 1 ~ 18 |
| Sqrt | 1 ~ 18 |
| Square | 1 ~ 18 |
| SquaredDifference | 1 ~ 18 |
| SquaredDistance | 12 ~ 18 |
| Squeeze | 1 ~ 18 |
| StatelessIf | 1 ~ 18 |
| StatelessWhile | 7 ~ 18 |
| StopGradient | 1 ~ 18 |
| StridedSlice | 1 ~ 18 |
| StringLower | 10 ~ 18 |
| StringToNumber | 9 ~ 18 |
| StringUpper | 10 ~ 18 |
| Sub | 1 ~ 18 |
| Sum | 1 ~ 18 |
| TFL_CONCATENATION | 1 ~ 18 |
| TFL_DEQUANTIZE | 1 ~ 18 |
| TFL_PRELU | 7 ~ 18 |
| TFL_QUANTIZE | 1 ~ 18 |
| TFL_TFLite_Detection_PostProcess | 11 ~ 18 |
| TFL_WHILE | 7 ~ 18 |
| Tan | 7 ~ 18 |
| Tanh | 1 ~ 18 |
| TensorListFromTensor | 7 ~ 18 |
| TensorListGetItem | 7 ~ 18 |
| TensorListLength | 7 ~ 18 |
| TensorListReserve | 7 ~ 18 |
| TensorListResize | 7 ~ 18 |
| TensorListSetItem | 7 ~ 18 |
| TensorListStack | 7 ~ 18 |
| TensorScatterAdd | 16 ~ 18 |
| TensorScatterMax | 16 ~ 18 |
| TensorScatterMin | 16 ~ 18 |
| TensorScatterSub | 16 ~ 18 |
| TensorScatterUpdate | 11 ~ 18 |
| Tile | 1 ~ 18 |
| TopKV2 | 1 ~ 18 |
| Transpose | 1 ~ 18 |
| TruncateDiv | 1 ~ 18 |
| Unique | 11 ~ 18 |
| UniqueWithCounts | 11 ~ 18 |
| Unpack | 1 ~ 18 |
| UnsortedSegmentMax | 11 ~ 18 |
| UnsortedSegmentMin | 11 ~ 18 |
| UnsortedSegmentProd | 11 ~ 18 |
| UnsortedSegmentSum | 11 ~ 18 |
| Where | 9 ~ 18 |
| While | 7 ~ 18 |
| ZerosLike | 1 ~ 18 |
### Domain: "com.google.tensorflow" 
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
### Domain: "com.microsoft" 
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Conv2DBackpropInput | 1 |
| CropAndResize | 1 |
| MatrixInverse | 1 |
| Range | 1 |
### Domain: "ai.onnx.contrib" 
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Equal | 1 |
| NotEqual | 1 |
| RegexSplitWithOffsets | 1 |
| SentencepieceOp | 1 |
| SentencepieceTokenizeOp | 1 |
| StaticRegexReplace | 1 |
| StringJoin | 1 |
| StringSplit | 1 |
| StringSplitV2 | 1 |
| StringToHashBucketFast | 1 |
| WordpieceTokenizeWithOffsets | 1 |
