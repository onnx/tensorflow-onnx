<!--- SPDX-License-Identifier: Apache-2.0 -->

## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 15 |
| Acos | 7 ~ 15 |
| Acosh | 9 ~ 15 |
| Add | 1 ~ 15 |
| AddN | 6 ~ 15 |
| AddV2 | 1 ~ 15 |
| AdjustContrastv2 | 1 ~ 15 |
| AdjustHue | 11 ~ 15 |
| AdjustSaturation | 11 ~ 15 |
| All | 6 ~ 15 |
| Any | 6 ~ 15 |
| ArgMax | 1 ~ 15 |
| ArgMin | 1 ~ 15 |
| AsString | 9 ~ 15 |
| Asin | 7 ~ 15 |
| Asinh | 9 ~ 15 |
| Atan | 7 ~ 15 |
| Atan2 | 9 ~ 15 |
| Atanh | 9 ~ 15 |
| AvgPool | 1 ~ 15 |
| AvgPool3D | 1 ~ 15 |
| BatchMatMul | 1 ~ 15 |
| BatchMatMulV2 | 1 ~ 15 |
| BatchToSpaceND | 1 ~ 15 |
| BiasAdd | 1 ~ 15 |
| BiasAddV1 | 1 ~ 15 |
| Bincount | 11 ~ 15 |
| BroadcastTo | 8 ~ 15 |
| CTCGreedyDecoder | 11 ~ 15 |
| Cast | 1 ~ 15 |
| Ceil | 1 ~ 15 |
| CheckNumerics | 1 ~ 15 |
| ClipByValue | 8 ~ 15 |
| CombinedNonMaxSuppression | 12 ~ 15 |
| ComplexAbs | 1 ~ 15 |
| Concat | 1 ~ 15 |
| ConcatV2 | 1 ~ 15 |
| Const | 1 ~ 15 |
| ConstV2 | 1 ~ 15 |
| Conv1D | 1 ~ 15 |
| Conv2D | 1 ~ 15 |
| Conv2DBackpropInput | 1 ~ 15 |
| Conv3D | 1 ~ 15 |
| Conv3DBackpropInputV2 | 1 ~ 15 |
| Cos | 7 ~ 15 |
| Cosh | 9 ~ 15 |
| CropAndResize | 10 ~ 15 |
| CudnnRNN | 10 ~ 15 |
| Cumsum | 11 ~ 15 |
| DenseBincount | 11 ~ 15 |
| DenseToDenseSetOperation | 11 ~ 15 |
| DepthToSpace | 1 ~ 15 |
| DepthwiseConv2d | 1 ~ 15 |
| DepthwiseConv2dNative | 1 ~ 15 |
| Div | 1 ~ 15 |
| DivNoNan | 9 ~ 15 |
| Dropout | 1 ~ 15 |
| DynamicPartition | 9 ~ 15 |
| DynamicStitch | 10 ~ 15 |
| Einsum | 12 ~ 15 |
| Elu | 1 ~ 15 |
| EnsureShape | 1 ~ 15 |
| Equal | 1 ~ 15 |
| Erf | 1 ~ 15 |
| Exp | 1 ~ 15 |
| ExpandDims | 1 ~ 15 |
| FFT | 1 ~ 15 |
| FIFOQueueV2 | 8 ~ 15 |
| FakeQuantWithMinMaxArgs | 10 ~ 15 |
| FakeQuantWithMinMaxVars | 10 ~ 15 |
| Fill | 7 ~ 15 |
| Flatten | 1 ~ 15 |
| Floor | 1 ~ 15 |
| FloorDiv | 6 ~ 15 |
| FloorMod | 7 ~ 15 |
| FusedBatchNorm | 6 ~ 15 |
| FusedBatchNormV2 | 6 ~ 15 |
| FusedBatchNormV3 | 6 ~ 15 |
| Gather | 1 ~ 15 |
| GatherNd | 1 ~ 15 |
| GatherV2 | 1 ~ 15 |
| Greater | 1 ~ 15 |
| GreaterEqual | 7 ~ 15 |
| HardSwish | 14 ~ 15 |
| HashTableV2 | 8 ~ 15 |
| Identity | 1 ~ 15 |
| IdentityN | 1 ~ 15 |
| If | 1 ~ 15 |
| InvertPermutation | 11 ~ 15 |
| IsFinite | 10 ~ 15 |
| IsInf | 10 ~ 15 |
| IsNan | 9 ~ 15 |
| IteratorGetNext | 8 ~ 15 |
| IteratorV2 | 8 ~ 15 |
| LRN | 1 ~ 15 |
| LSTMBlockCell | 1 ~ 15 |
| LeakyRelu | 1 ~ 15 |
| LeftShift | 11 ~ 15 |
| Less | 1 ~ 15 |
| LessEqual | 7 ~ 15 |
| Log | 1 ~ 15 |
| LogSoftmax | 1 ~ 15 |
| LogicalAnd | 1 ~ 15 |
| LogicalNot | 1 ~ 15 |
| LogicalOr | 1 ~ 15 |
| LookupTableFindV2 | 8 ~ 15 |
| LookupTableSizeV2 | 1 ~ 15 |
| Loop | 7 ~ 15 |
| MatMul | 1 ~ 15 |
| MatrixBandPart | 7 ~ 15 |
| MatrixDeterminant | 11 ~ 15 |
| MatrixDiag | 12 ~ 15 |
| MatrixDiagPart | 11 ~ 15 |
| MatrixDiagPartV2 | 11 ~ 15 |
| MatrixDiagPartV3 | 11 ~ 15 |
| MatrixDiagV2 | 12 ~ 15 |
| MatrixDiagV3 | 12 ~ 15 |
| MatrixSetDiagV3 | 12 ~ 15 |
| Max | 1 ~ 15 |
| MaxPool | 1 ~ 15 |
| MaxPool3D | 1 ~ 15 |
| MaxPoolV2 | 1 ~ 15 |
| MaxPoolWithArgmax | 8 ~ 15 |
| Maximum | 1 ~ 15 |
| Mean | 1 ~ 15 |
| Min | 1 ~ 15 |
| Minimum | 1 ~ 15 |
| MirrorPad | 1 ~ 15 |
| Mul | 1 ~ 15 |
| Multinomial | 7 ~ 15 |
| Neg | 1 ~ 15 |
| NoOp | 1 ~ 15 |
| NonMaxSuppressionV2 | 10 ~ 15 |
| NonMaxSuppressionV3 | 10 ~ 15 |
| NonMaxSuppressionV4 | 10 ~ 15 |
| NonMaxSuppressionV5 | 10 ~ 15 |
| NotEqual | 1 ~ 15 |
| OneHot | 1 ~ 15 |
| Pack | 1 ~ 15 |
| Pad | 1 ~ 15 |
| PadV2 | 1 ~ 15 |
| ParallelDynamicStitch | 10 ~ 15 |
| Placeholder | 1 ~ 15 |
| PlaceholderV2 | 1 ~ 15 |
| PlaceholderWithDefault | 1 ~ 15 |
| Pow | 1 ~ 15 |
| Prelu | 1 ~ 15 |
| Prod | 1 ~ 15 |
| QueueDequeueManyV2 | 8 ~ 15 |
| QueueDequeueUpToV2 | 8 ~ 15 |
| QueueDequeueV2 | 8 ~ 15 |
| RFFT | 1 ~ 15 |
| RFFT2D | 1 ~ 15 |
| RaggedGather | 11 ~ 15 |
| RaggedRange | 11 ~ 15 |
| RaggedTensorFromVariant | 13 ~ 15 |
| RaggedTensorToSparse | 11 ~ 15 |
| RaggedTensorToTensor | 11 ~ 15 |
| RaggedTensorToVariant | 13 ~ 15 |
| RandomNormal | 1 ~ 15 |
| RandomNormalLike | 1 ~ 15 |
| RandomShuffle | 10 ~ 15 |
| RandomStandardNormal | 1 ~ 15 |
| RandomUniform | 1 ~ 15 |
| RandomUniformInt | 1 ~ 15 |
| RandomUniformLike | 1 ~ 15 |
| Range | 7 ~ 15 |
| RealDiv | 1 ~ 15 |
| Reciprocal | 1 ~ 15 |
| Relu | 1 ~ 15 |
| Relu6 | 1 ~ 15 |
| Reshape | 1 ~ 15 |
| ResizeBicubic | 7 ~ 15 |
| ResizeBilinear | 7 ~ 15 |
| ResizeNearestNeighbor | 7 ~ 15 |
| ReverseSequence | 8 ~ 15 (Except 9) |
| ReverseV2 | 10 ~ 15 |
| RightShift | 11 ~ 15 |
| Roll | 10 ~ 15 |
| Round | 1 ~ 15 |
| Rsqrt | 1 ~ 15 |
| SampleDistortedBoundingBox | 9 ~ 15 |
| SampleDistortedBoundingBoxV2 | 9 ~ 15 |
| Scan | 7 ~ 15 |
| ScatterNd | 11 ~ 15 |
| SegmentMax | 11 ~ 15 |
| SegmentMean | 11 ~ 15 |
| SegmentMin | 11 ~ 15 |
| SegmentProd | 11 ~ 15 |
| SegmentSum | 11 ~ 15 |
| Select | 7 ~ 15 |
| SelectV2 | 7 ~ 15 |
| Selu | 1 ~ 15 |
| Shape | 1 ~ 15 |
| Sigmoid | 1 ~ 15 |
| Sign | 1 ~ 15 |
| Sin | 7 ~ 15 |
| Sinh | 9 ~ 15 |
| Size | 1 ~ 15 |
| Slice | 1 ~ 15 |
| Softmax | 1 ~ 15 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 15 |
| Softplus | 1 ~ 15 |
| Softsign | 1 ~ 15 |
| SpaceToBatchND | 1 ~ 15 |
| SpaceToDepth | 1 ~ 15 |
| SparseFillEmptyRows | 11 ~ 15 |
| SparseReshape | 11 ~ 15 |
| SparseSegmentMean | 11 ~ 15 |
| SparseSegmentMeanWithNumSegments | 11 ~ 15 |
| SparseSegmentSqrtN | 11 ~ 15 |
| SparseSegmentSqrtNWithNumSegments | 11 ~ 15 |
| SparseSegmentSum | 11 ~ 15 |
| SparseSegmentSumWithNumSegments | 11 ~ 15 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 15 |
| SparseToDense | 11 ~ 15 |
| Split | 1 ~ 15 |
| SplitV | 1 ~ 15 |
| Sqrt | 1 ~ 15 |
| Square | 1 ~ 15 |
| SquaredDifference | 1 ~ 15 |
| SquaredDistance | 12 ~ 15 |
| Squeeze | 1 ~ 15 |
| StatelessIf | 1 ~ 15 |
| StatelessWhile | 7 ~ 15 |
| StopGradient | 1 ~ 15 |
| StridedSlice | 1 ~ 15 |
| StringLower | 10 ~ 15 |
| StringToNumber | 9 ~ 15 |
| StringUpper | 10 ~ 15 |
| Sub | 1 ~ 15 |
| Sum | 1 ~ 15 |
| TFL_CONCATENATION | 1 ~ 15 |
| TFL_DEQUANTIZE | 1 ~ 15 |
| TFL_PRELU | 7 ~ 15 |
| TFL_QUANTIZE | 1 ~ 15 |
| TFL_TFLite_Detection_PostProcess | 11 ~ 15 |
| TFL_WHILE | 7 ~ 15 |
| Tan | 7 ~ 15 |
| Tanh | 1 ~ 15 |
| TensorListFromTensor | 7 ~ 15 |
| TensorListGetItem | 7 ~ 15 |
| TensorListLength | 7 ~ 15 |
| TensorListReserve | 7 ~ 15 |
| TensorListResize | 7 ~ 15 |
| TensorListSetItem | 7 ~ 15 |
| TensorListStack | 7 ~ 15 |
| TensorScatterUpdate | 11 ~ 15 |
| Tile | 1 ~ 15 |
| TopKV2 | 1 ~ 15 |
| Transpose | 1 ~ 15 |
| TruncateDiv | 1 ~ 15 |
| Unique | 11 ~ 15 |
| Unpack | 1 ~ 15 |
| UnsortedSegmentMax | 11 ~ 15 |
| UnsortedSegmentMin | 11 ~ 15 |
| UnsortedSegmentProd | 11 ~ 15 |
| UnsortedSegmentSum | 11 ~ 15 |
| Where | 9 ~ 15 |
| While | 7 ~ 15 |
| ZerosLike | 1 ~ 15 |
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
