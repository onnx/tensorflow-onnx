<!--- SPDX-License-Identifier: Apache-2.0 -->

## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 16 |
| Acos | 7 ~ 16 |
| Acosh | 9 ~ 16 |
| Add | 1 ~ 16 |
| AddN | 6 ~ 16 |
| AddV2 | 1 ~ 16 |
| AdjustContrastv2 | 1 ~ 16 |
| AdjustHue | 11 ~ 16 |
| AdjustSaturation | 11 ~ 16 |
| All | 6 ~ 16 |
| Any | 6 ~ 16 |
| ArgMax | 1 ~ 16 |
| ArgMin | 1 ~ 16 |
| AsString | 9 ~ 16 |
| Asin | 7 ~ 16 |
| Asinh | 9 ~ 16 |
| Atan | 7 ~ 16 |
| Atan2 | 9 ~ 16 |
| Atanh | 9 ~ 16 |
| AvgPool | 1 ~ 16 |
| AvgPool3D | 1 ~ 16 |
| BatchMatMul | 1 ~ 16 |
| BatchMatMulV2 | 1 ~ 16 |
| BatchToSpaceND | 1 ~ 16 |
| BiasAdd | 1 ~ 16 |
| BiasAddV1 | 1 ~ 16 |
| Bincount | 11 ~ 16 |
| BroadcastTo | 8 ~ 16 |
| CTCGreedyDecoder | 11 ~ 16 |
| Cast | 1 ~ 16 |
| Ceil | 1 ~ 16 |
| CheckNumerics | 1 ~ 16 |
| ClipByValue | 8 ~ 16 |
| CombinedNonMaxSuppression | 12 ~ 16 |
| ComplexAbs | 1 ~ 16 |
| Concat | 1 ~ 16 |
| ConcatV2 | 1 ~ 16 |
| Const | 1 ~ 16 |
| ConstV2 | 1 ~ 16 |
| Conv1D | 1 ~ 16 |
| Conv2D | 1 ~ 16 |
| Conv2DBackpropInput | 1 ~ 16 |
| Conv3D | 1 ~ 16 |
| Conv3DBackpropInputV2 | 1 ~ 16 |
| Cos | 7 ~ 16 |
| Cosh | 9 ~ 16 |
| CropAndResize | 10 ~ 16 |
| CudnnRNN | 10 ~ 16 |
| Cumsum | 11 ~ 16 |
| DenseBincount | 11 ~ 16 |
| DenseToDenseSetOperation | 11 ~ 16 |
| DepthToSpace | 1 ~ 16 |
| DepthwiseConv2d | 1 ~ 16 |
| DepthwiseConv2dNative | 1 ~ 16 |
| Div | 1 ~ 16 |
| DivNoNan | 9 ~ 16 |
| Dropout | 1 ~ 16 |
| DynamicPartition | 9 ~ 16 |
| DynamicStitch | 10 ~ 16 |
| Einsum | 12 ~ 16 |
| Elu | 1 ~ 16 |
| EnsureShape | 1 ~ 16 |
| Equal | 1 ~ 16 |
| Erf | 1 ~ 16 |
| Exp | 1 ~ 16 |
| ExpandDims | 1 ~ 16 |
| FFT | 1 ~ 16 |
| FIFOQueueV2 | 8 ~ 16 |
| FakeQuantWithMinMaxArgs | 10 ~ 16 |
| FakeQuantWithMinMaxVars | 10 ~ 16 |
| Fill | 7 ~ 16 |
| Flatten | 1 ~ 16 |
| Floor | 1 ~ 16 |
| FloorDiv | 6 ~ 16 |
| FloorMod | 7 ~ 16 |
| FusedBatchNorm | 6 ~ 16 |
| FusedBatchNormV2 | 6 ~ 16 |
| FusedBatchNormV3 | 6 ~ 16 |
| Gather | 1 ~ 16 |
| GatherNd | 1 ~ 16 |
| GatherV2 | 1 ~ 16 |
| Greater | 1 ~ 16 |
| GreaterEqual | 7 ~ 16 |
| HardSwish | 14 ~ 16 |
| HashTableV2 | 8 ~ 16 |
| Identity | 1 ~ 16 |
| IdentityN | 1 ~ 16 |
| If | 1 ~ 16 |
| InvertPermutation | 11 ~ 16 |
| IsFinite | 10 ~ 16 |
| IsInf | 10 ~ 16 |
| IsNan | 9 ~ 16 |
| IteratorGetNext | 8 ~ 16 |
| IteratorV2 | 8 ~ 16 |
| LRN | 1 ~ 16 |
| LSTMBlockCell | 1 ~ 16 |
| LeakyRelu | 1 ~ 16 |
| LeftShift | 11 ~ 16 |
| Less | 1 ~ 16 |
| LessEqual | 7 ~ 16 |
| Log | 1 ~ 16 |
| LogSoftmax | 1 ~ 16 |
| LogicalAnd | 1 ~ 16 |
| LogicalNot | 1 ~ 16 |
| LogicalOr | 1 ~ 16 |
| LookupTableFindV2 | 8 ~ 16 |
| LookupTableSizeV2 | 1 ~ 16 |
| Loop | 7 ~ 16 |
| MatMul | 1 ~ 16 |
| MatrixBandPart | 7 ~ 16 |
| MatrixDeterminant | 11 ~ 16 |
| MatrixDiag | 12 ~ 16 |
| MatrixDiagPart | 11 ~ 16 |
| MatrixDiagPartV2 | 11 ~ 16 |
| MatrixDiagPartV3 | 11 ~ 16 |
| MatrixDiagV2 | 12 ~ 16 |
| MatrixDiagV3 | 12 ~ 16 |
| MatrixSetDiagV3 | 12 ~ 16 |
| Max | 1 ~ 16 |
| MaxPool | 1 ~ 16 |
| MaxPool3D | 1 ~ 16 |
| MaxPoolV2 | 1 ~ 16 |
| MaxPoolWithArgmax | 8 ~ 16 |
| Maximum | 1 ~ 16 |
| Mean | 1 ~ 16 |
| Min | 1 ~ 16 |
| Minimum | 1 ~ 16 |
| MirrorPad | 1 ~ 16 |
| Mul | 1 ~ 16 |
| Multinomial | 7 ~ 16 |
| Neg | 1 ~ 16 |
| NoOp | 1 ~ 16 |
| NonMaxSuppressionV2 | 10 ~ 16 |
| NonMaxSuppressionV3 | 10 ~ 16 |
| NonMaxSuppressionV4 | 10 ~ 16 |
| NonMaxSuppressionV5 | 10 ~ 16 |
| NotEqual | 1 ~ 16 |
| OneHot | 1 ~ 16 |
| Pack | 1 ~ 16 |
| Pad | 1 ~ 16 |
| PadV2 | 1 ~ 16 |
| ParallelDynamicStitch | 10 ~ 16 |
| Placeholder | 1 ~ 16 |
| PlaceholderV2 | 1 ~ 16 |
| PlaceholderWithDefault | 1 ~ 16 |
| Pow | 1 ~ 16 |
| Prelu | 1 ~ 16 |
| Prod | 1 ~ 16 |
| QueueDequeueManyV2 | 8 ~ 16 |
| QueueDequeueUpToV2 | 8 ~ 16 |
| QueueDequeueV2 | 8 ~ 16 |
| RFFT | 1 ~ 16 |
| RFFT2D | 1 ~ 16 |
| RaggedGather | 11 ~ 16 |
| RaggedRange | 11 ~ 16 |
| RaggedTensorFromVariant | 13 ~ 16 |
| RaggedTensorToSparse | 11 ~ 16 |
| RaggedTensorToTensor | 11 ~ 16 |
| RaggedTensorToVariant | 13 ~ 16 |
| RandomNormal | 1 ~ 16 |
| RandomNormalLike | 1 ~ 16 |
| RandomShuffle | 10 ~ 16 |
| RandomStandardNormal | 1 ~ 16 |
| RandomUniform | 1 ~ 16 |
| RandomUniformInt | 1 ~ 16 |
| RandomUniformLike | 1 ~ 16 |
| Range | 7 ~ 16 |
| RealDiv | 1 ~ 16 |
| Reciprocal | 1 ~ 16 |
| Relu | 1 ~ 16 |
| Relu6 | 1 ~ 16 |
| Reshape | 1 ~ 16 |
| ResizeBicubic | 7 ~ 16 |
| ResizeBilinear | 7 ~ 16 |
| ResizeNearestNeighbor | 7 ~ 16 |
| ReverseSequence | 8 ~ 16 (Except 9) |
| ReverseV2 | 10 ~ 16 |
| RightShift | 11 ~ 16 |
| Rint | 11 ~ 16 |
| Roll | 10 ~ 16 |
| Round | 1 ~ 16 |
| Rsqrt | 1 ~ 16 |
| SampleDistortedBoundingBox | 9 ~ 16 |
| SampleDistortedBoundingBoxV2 | 9 ~ 16 |
| Scan | 7 ~ 16 |
| ScatterNd | 11 ~ 16 |
| SegmentMax | 11 ~ 16 |
| SegmentMean | 11 ~ 16 |
| SegmentMin | 11 ~ 16 |
| SegmentProd | 11 ~ 16 |
| SegmentSum | 11 ~ 16 |
| Select | 7 ~ 16 |
| SelectV2 | 7 ~ 16 |
| Selu | 1 ~ 16 |
| Shape | 1 ~ 16 |
| Sigmoid | 1 ~ 16 |
| Sign | 1 ~ 16 |
| Sin | 7 ~ 16 |
| Sinh | 9 ~ 16 |
| Size | 1 ~ 16 |
| Slice | 1 ~ 16 |
| Softmax | 1 ~ 16 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 16 |
| Softplus | 1 ~ 16 |
| Softsign | 1 ~ 16 |
| SpaceToBatchND | 1 ~ 16 |
| SpaceToDepth | 1 ~ 16 |
| SparseFillEmptyRows | 11 ~ 16 |
| SparseReshape | 11 ~ 16 |
| SparseSegmentMean | 11 ~ 16 |
| SparseSegmentMeanWithNumSegments | 11 ~ 16 |
| SparseSegmentSqrtN | 11 ~ 16 |
| SparseSegmentSqrtNWithNumSegments | 11 ~ 16 |
| SparseSegmentSum | 11 ~ 16 |
| SparseSegmentSumWithNumSegments | 11 ~ 16 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 16 |
| SparseToDense | 11 ~ 16 |
| Split | 1 ~ 16 |
| SplitV | 1 ~ 16 |
| Sqrt | 1 ~ 16 |
| Square | 1 ~ 16 |
| SquaredDifference | 1 ~ 16 |
| SquaredDistance | 12 ~ 16 |
| Squeeze | 1 ~ 16 |
| StatelessIf | 1 ~ 16 |
| StatelessWhile | 7 ~ 16 |
| StopGradient | 1 ~ 16 |
| StridedSlice | 1 ~ 16 |
| StringLower | 10 ~ 16 |
| StringToNumber | 9 ~ 16 |
| StringUpper | 10 ~ 16 |
| Sub | 1 ~ 16 |
| Sum | 1 ~ 16 |
| TFL_CONCATENATION | 1 ~ 16 |
| TFL_DEQUANTIZE | 1 ~ 16 |
| TFL_PRELU | 7 ~ 16 |
| TFL_QUANTIZE | 1 ~ 16 |
| TFL_TFLite_Detection_PostProcess | 11 ~ 16 |
| TFL_WHILE | 7 ~ 16 |
| Tan | 7 ~ 16 |
| Tanh | 1 ~ 16 |
| TensorListFromTensor | 7 ~ 16 |
| TensorListGetItem | 7 ~ 16 |
| TensorListLength | 7 ~ 16 |
| TensorListReserve | 7 ~ 16 |
| TensorListResize | 7 ~ 16 |
| TensorListSetItem | 7 ~ 16 |
| TensorListStack | 7 ~ 16 |
| TensorScatterAdd | 16 |
| TensorScatterUpdate | 11 ~ 16 |
| Tile | 1 ~ 16 |
| TopKV2 | 1 ~ 16 |
| Transpose | 1 ~ 16 |
| TruncateDiv | 1 ~ 16 |
| Unique | 11 ~ 16 |
| Unpack | 1 ~ 16 |
| UnsortedSegmentMax | 11 ~ 16 |
| UnsortedSegmentMin | 11 ~ 16 |
| UnsortedSegmentProd | 11 ~ 16 |
| UnsortedSegmentSum | 11 ~ 16 |
| Where | 9 ~ 16 |
| While | 7 ~ 16 |
| ZerosLike | 1 ~ 16 |
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
