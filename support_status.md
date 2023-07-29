<!--- SPDX-License-Identifier: Apache-2.0 -->

## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 17 |
| Acos | 7 ~ 17 |
| Acosh | 9 ~ 17 |
| Add | 1 ~ 17 |
| AddN | 6 ~ 17 |
| AddV2 | 1 ~ 17 |
| AdjustContrastv2 | 1 ~ 17 |
| AdjustHue | 11 ~ 17 |
| AdjustSaturation | 11 ~ 17 |
| All | 6 ~ 17 |
| Any | 6 ~ 17 |
| ArgMax | 1 ~ 17 |
| ArgMin | 1 ~ 17 |
| AsString | 9 ~ 17 |
| Asin | 7 ~ 17 |
| Asinh | 9 ~ 17 |
| Atan | 7 ~ 17 |
| Atan2 | 9 ~ 17 |
| Atanh | 9 ~ 17 |
| AvgPool | 1 ~ 17 |
| AvgPool3D | 1 ~ 17 |
| BatchMatMul | 1 ~ 17 |
| BatchMatMulV2 | 1 ~ 17 |
| BatchMatMulV3 | 1 ~ 17 |
| BatchToSpaceND | 1 ~ 17 |
| BiasAdd | 1 ~ 17 |
| BiasAddV1 | 1 ~ 17 |
| Bincount | 11 ~ 17 |
| BitwiseAnd | 18 |
| BitwiseOr | 18 |
| BitwiseXor | 18 |
| BroadcastTo | 8 ~ 17 |
| CTCGreedyDecoder | 11 ~ 17 |
| Cast | 1 ~ 17 |
| Ceil | 1 ~ 17 |
| CheckNumerics | 1 ~ 17 |
| ClipByValue | 8 ~ 17 |
| CombinedNonMaxSuppression | 12 ~ 17 |
| ComplexAbs | 1 ~ 17 |
| Concat | 1 ~ 17 |
| ConcatV2 | 1 ~ 17 |
| Const | 1 ~ 17 |
| ConstV2 | 1 ~ 17 |
| Conv1D | 1 ~ 17 |
| Conv2D | 1 ~ 17 |
| Conv2DBackpropInput | 1 ~ 17 |
| Conv3D | 1 ~ 17 |
| Conv3DBackpropInputV2 | 1 ~ 17 |
| Cos | 7 ~ 17 |
| Cosh | 9 ~ 17 |
| CropAndResize | 10 ~ 17 |
| CudnnRNN | 10 ~ 17 |
| Cumsum | 11 ~ 17 |
| DenseBincount | 11 ~ 17 |
| DenseToDenseSetOperation | 11 ~ 17 |
| DepthToSpace | 1 ~ 17 |
| DepthwiseConv2d | 1 ~ 17 |
| DepthwiseConv2dNative | 1 ~ 17 |
| Div | 1 ~ 17 |
| DivNoNan | 9 ~ 17 |
| Dropout | 1 ~ 17 |
| DynamicPartition | 9 ~ 17 |
| DynamicStitch | 10 ~ 17 |
| Einsum | 12 ~ 17 |
| Elu | 1 ~ 17 |
| EnsureShape | 1 ~ 17 |
| Equal | 1 ~ 17 |
| Erf | 1 ~ 17 |
| Exp | 1 ~ 17 |
| ExpandDims | 1 ~ 17 |
| FFT | 1 ~ 17 |
| FIFOQueueV2 | 8 ~ 17 |
| FakeQuantWithMinMaxArgs | 10 ~ 17 |
| FakeQuantWithMinMaxVars | 10 ~ 17 |
| Fill | 7 ~ 17 |
| Flatten | 1 ~ 17 |
| Floor | 1 ~ 17 |
| FloorDiv | 6 ~ 17 |
| FloorMod | 7 ~ 17 |
| FusedBatchNorm | 6 ~ 17 |
| FusedBatchNormV2 | 6 ~ 17 |
| FusedBatchNormV3 | 6 ~ 17 |
| Gather | 1 ~ 17 |
| GatherNd | 1 ~ 17 |
| GatherV2 | 1 ~ 17 |
| Greater | 1 ~ 17 |
| GreaterEqual | 7 ~ 17 |
| HardSwish | 14 ~ 17 |
| HashTableV2 | 8 ~ 17 |
| Identity | 1 ~ 17 |
| IdentityN | 1 ~ 17 |
| If | 1 ~ 17 |
| Invert | 18 |
| InvertPermutation | 11 ~ 17 |
| IsFinite | 10 ~ 17 |
| IsInf | 10 ~ 17 |
| IsNan | 9 ~ 17 |
| IteratorGetNext | 8 ~ 17 |
| IteratorV2 | 8 ~ 17 |
| LRN | 1 ~ 17 |
| LSTMBlockCell | 1 ~ 17 |
| LeakyRelu | 1 ~ 17 |
| LeftShift | 11 ~ 17 |
| Less | 1 ~ 17 |
| LessEqual | 7 ~ 17 |
| Log | 1 ~ 17 |
| LogSoftmax | 1 ~ 17 |
| LogicalAnd | 1 ~ 17 |
| LogicalNot | 1 ~ 17 |
| LogicalOr | 1 ~ 17 |
| LookupTableFindV2 | 8 ~ 17 |
| LookupTableSizeV2 | 1 ~ 17 |
| Loop | 7 ~ 17 |
| MatMul | 1 ~ 17 |
| MatrixBandPart | 7 ~ 17 |
| MatrixDeterminant | 11 ~ 17 |
| MatrixDiag | 12 ~ 17 |
| MatrixDiagPart | 11 ~ 17 |
| MatrixDiagPartV2 | 11 ~ 17 |
| MatrixDiagPartV3 | 11 ~ 17 |
| MatrixDiagV2 | 12 ~ 17 |
| MatrixDiagV3 | 12 ~ 17 |
| MatrixSetDiagV3 | 12 ~ 17 |
| Max | 1 ~ 17 |
| MaxPool | 1 ~ 17 |
| MaxPool3D | 1 ~ 17 |
| MaxPoolV2 | 1 ~ 17 |
| MaxPoolWithArgmax | 8 ~ 17 |
| Maximum | 1 ~ 17 |
| Mean | 1 ~ 17 |
| Min | 1 ~ 17 |
| Minimum | 1 ~ 17 |
| MirrorPad | 1 ~ 17 |
| Mul | 1 ~ 17 |
| Multinomial | 7 ~ 17 |
| Neg | 1 ~ 17 |
| NoOp | 1 ~ 17 |
| NonMaxSuppressionV2 | 10 ~ 17 |
| NonMaxSuppressionV3 | 10 ~ 17 |
| NonMaxSuppressionV4 | 10 ~ 17 |
| NonMaxSuppressionV5 | 10 ~ 17 |
| NotEqual | 1 ~ 17 |
| OneHot | 1 ~ 17 |
| Pack | 1 ~ 17 |
| Pad | 1 ~ 17 |
| PadV2 | 1 ~ 17 |
| ParallelDynamicStitch | 10 ~ 17 |
| Placeholder | 1 ~ 17 |
| PlaceholderV2 | 1 ~ 17 |
| PlaceholderWithDefault | 1 ~ 17 |
| Pow | 1 ~ 17 |
| Prelu | 1 ~ 17 |
| Prod | 1 ~ 17 |
| QueueDequeueManyV2 | 8 ~ 17 |
| QueueDequeueUpToV2 | 8 ~ 17 |
| QueueDequeueV2 | 8 ~ 17 |
| RFFT | 1 ~ 17 |
| RFFT2D | 1 ~ 17 |
| RaggedGather | 11 ~ 17 |
| RaggedRange | 11 ~ 17 |
| RaggedTensorFromVariant | 13 ~ 17 |
| RaggedTensorToSparse | 11 ~ 17 |
| RaggedTensorToTensor | 11 ~ 17 |
| RaggedTensorToVariant | 13 ~ 17 |
| RandomNormal | 1 ~ 17 |
| RandomNormalLike | 1 ~ 17 |
| RandomShuffle | 10 ~ 17 |
| RandomStandardNormal | 1 ~ 17 |
| RandomUniform | 1 ~ 17 |
| RandomUniformInt | 1 ~ 17 |
| RandomUniformLike | 1 ~ 17 |
| Range | 7 ~ 17 |
| RealDiv | 1 ~ 17 |
| Reciprocal | 1 ~ 17 |
| Relu | 1 ~ 17 |
| Relu6 | 1 ~ 17 |
| Reshape | 1 ~ 17 |
| ResizeBicubic | 7 ~ 17 |
| ResizeBilinear | 7 ~ 17 |
| ResizeNearestNeighbor | 7 ~ 17 |
| ReverseSequence | 8 ~ 17 (Except 9) |
| ReverseV2 | 10 ~ 17 |
| RightShift | 11 ~ 17 |
| Rint | 11 ~ 17 |
| Roll | 10 ~ 17 |
| Round | 1 ~ 17 |
| Rsqrt | 1 ~ 17 |
| SampleDistortedBoundingBox | 9 ~ 17 |
| SampleDistortedBoundingBoxV2 | 9 ~ 17 |
| Scan | 7 ~ 17 |
| ScatterNd | 11 ~ 17 |
| SegmentMax | 11 ~ 17 |
| SegmentMean | 11 ~ 17 |
| SegmentMin | 11 ~ 17 |
| SegmentProd | 11 ~ 17 |
| SegmentSum | 11 ~ 17 |
| Select | 7 ~ 17 |
| SelectV2 | 7 ~ 17 |
| Selu | 1 ~ 17 |
| Shape | 1 ~ 17 |
| Sigmoid | 1 ~ 17 |
| Sign | 1 ~ 17 |
| Sin | 7 ~ 17 |
| Sinh | 9 ~ 17 |
| Size | 1 ~ 17 |
| Slice | 1 ~ 17 |
| Softmax | 1 ~ 17 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 17 |
| Softplus | 1 ~ 17 |
| Softsign | 1 ~ 17 |
| SpaceToBatchND | 1 ~ 17 |
| SpaceToDepth | 1 ~ 17 |
| SparseFillEmptyRows | 11 ~ 17 |
| SparseReshape | 11 ~ 17 |
| SparseSegmentMean | 11 ~ 17 |
| SparseSegmentMeanWithNumSegments | 11 ~ 17 |
| SparseSegmentSqrtN | 11 ~ 17 |
| SparseSegmentSqrtNWithNumSegments | 11 ~ 17 |
| SparseSegmentSum | 11 ~ 17 |
| SparseSegmentSumWithNumSegments | 11 ~ 17 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 17 |
| SparseToDense | 11 ~ 17 |
| Split | 1 ~ 17 |
| SplitV | 1 ~ 17 |
| Sqrt | 1 ~ 17 |
| Square | 1 ~ 17 |
| SquaredDifference | 1 ~ 17 |
| SquaredDistance | 12 ~ 17 |
| Squeeze | 1 ~ 17 |
| StatelessIf | 1 ~ 17 |
| StatelessWhile | 7 ~ 17 |
| StopGradient | 1 ~ 17 |
| StridedSlice | 1 ~ 17 |
| StringLower | 10 ~ 17 |
| StringToNumber | 9 ~ 17 |
| StringUpper | 10 ~ 17 |
| Sub | 1 ~ 17 |
| Sum | 1 ~ 17 |
| TFL_CONCATENATION | 1 ~ 17 |
| TFL_DEQUANTIZE | 1 ~ 17 |
| TFL_PRELU | 7 ~ 17 |
| TFL_QUANTIZE | 1 ~ 17 |
| TFL_TFLite_Detection_PostProcess | 11 ~ 17 |
| TFL_WHILE | 7 ~ 17 |
| Tan | 7 ~ 17 |
| Tanh | 1 ~ 17 |
| TensorListFromTensor | 7 ~ 17 |
| TensorListGetItem | 7 ~ 17 |
| TensorListLength | 7 ~ 17 |
| TensorListReserve | 7 ~ 17 |
| TensorListResize | 7 ~ 17 |
| TensorListSetItem | 7 ~ 17 |
| TensorListStack | 7 ~ 17 |
| TensorScatterAdd | 16 ~ 17 |
| TensorScatterMax | 16 ~ 17 |
| TensorScatterMin | 16 ~ 17 |
| TensorScatterSub | 16 ~ 17 |
| TensorScatterUpdate | 11 ~ 17 |
| Tile | 1 ~ 17 |
| TopKV2 | 1 ~ 17 |
| Transpose | 1 ~ 17 |
| TruncateDiv | 1 ~ 17 |
| Unique | 11 ~ 17 |
| UniqueWithCounts | 11 ~ 18 |
| Unpack | 1 ~ 17 |
| UnsortedSegmentMax | 11 ~ 17 |
| UnsortedSegmentMin | 11 ~ 17 |
| UnsortedSegmentProd | 11 ~ 17 |
| UnsortedSegmentSum | 11 ~ 17 |
| Where | 9 ~ 17 |
| While | 7 ~ 17 |
| ZerosLike | 1 ~ 17 |
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
