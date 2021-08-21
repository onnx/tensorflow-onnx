## `tf2onnx` Support Status
### Domain: "" (default domain)
| Tensorflow Op | Convertible to ONNX Op Versions |
| ------------- | ------------------------------- |
| Abs | 1 ~ 14 |
| Acos | 7 ~ 14 |
| Acosh | 9 ~ 14 |
| Add | 1 ~ 14 |
| AddN | 6 ~ 14 |
| AddV2 | 1 ~ 14 |
| AdjustContrastv2 | 1 ~ 14 |
| AdjustHue | 11 ~ 14 |
| AdjustSaturation | 11 ~ 14 |
| All | 6 ~ 14 |
| Any | 6 ~ 14 |
| ArgMax | 1 ~ 14 |
| ArgMin | 1 ~ 14 |
| AsString | 9 ~ 14 |
| Asin | 7 ~ 14 |
| Asinh | 9 ~ 14 |
| Atan | 7 ~ 14 |
| Atan2 | 9 ~ 14 |
| Atanh | 9 ~ 14 |
| AvgPool | 1 ~ 14 |
| AvgPool3D | 1 ~ 14 |
| BatchMatMul | 1 ~ 14 |
| BatchMatMulV2 | 1 ~ 14 |
| BatchToSpaceND | 1 ~ 14 |
| BiasAdd | 1 ~ 14 |
| BiasAddV1 | 1 ~ 14 |
| Bincount | 11 ~ 14 |
| BroadcastTo | 8 ~ 14 |
| CTCGreedyDecoder | 11 ~ 14 |
| Cast | 1 ~ 14 |
| Ceil | 1 ~ 14 |
| CheckNumerics | 1 ~ 14 |
| ClipByValue | 8 ~ 14 |
| CombinedNonMaxSuppression | 12 ~ 14 |
| ComplexAbs | 1 ~ 14 |
| Concat | 1 ~ 14 |
| ConcatV2 | 1 ~ 14 |
| Const | 1 ~ 14 |
| ConstV2 | 1 ~ 14 |
| Conv1D | 1 ~ 14 |
| Conv2D | 1 ~ 14 |
| Conv2DBackpropInput | 1 ~ 14 |
| Conv3D | 1 ~ 14 |
| Conv3DBackpropInputV2 | 1 ~ 14 |
| Cos | 7 ~ 14 |
| Cosh | 9 ~ 14 |
| CropAndResize | 10 ~ 14 |
| CudnnRNN | 10 ~ 14 |
| Cumsum | 11 ~ 14 |
| DenseBincount | 11 ~ 14 |
| DenseToDenseSetOperation | 11 ~ 14 |
| DepthToSpace | 1 ~ 14 |
| DepthwiseConv2d | 1 ~ 14 |
| DepthwiseConv2dNative | 1 ~ 14 |
| Div | 1 ~ 14 |
| DivNoNan | 9 ~ 14 |
| Dropout | 1 ~ 14 |
| DynamicPartition | 9 ~ 14 |
| DynamicStitch | 10 ~ 14 |
| Einsum | 12 ~ 14 |
| Elu | 1 ~ 14 |
| EnsureShape | 1 ~ 14 |
| Equal | 1 ~ 14 |
| Erf | 1 ~ 14 |
| Exp | 1 ~ 14 |
| ExpandDims | 1 ~ 14 |
| FFT | 1 ~ 14 |
| FIFOQueueV2 | 8 ~ 14 |
| FakeQuantWithMinMaxArgs | 10 ~ 14 |
| FakeQuantWithMinMaxVars | 10 ~ 14 |
| Fill | 7 ~ 14 |
| Flatten | 1 ~ 14 |
| Floor | 1 ~ 14 |
| FloorDiv | 6 ~ 14 |
| FloorMod | 7 ~ 14 |
| FusedBatchNorm | 6 ~ 14 |
| FusedBatchNormV2 | 6 ~ 14 |
| FusedBatchNormV3 | 6 ~ 14 |
| Gather | 1 ~ 14 |
| GatherNd | 1 ~ 14 |
| GatherV2 | 1 ~ 14 |
| Greater | 1 ~ 14 |
| GreaterEqual | 7 ~ 14 |
| HashTableV2 | 8 ~ 14 |
| Identity | 1 ~ 14 |
| IdentityN | 1 ~ 14 |
| If | 1 ~ 14 |
| InvertPermutation | 11 ~ 14 |
| IsFinite | 10 ~ 14 |
| IsInf | 10 ~ 14 |
| IsNan | 9 ~ 14 |
| IteratorGetNext | 8 ~ 14 |
| IteratorV2 | 8 ~ 14 |
| LRN | 1 ~ 14 |
| LSTMBlockCell | 1 ~ 14 |
| LeakyRelu | 1 ~ 14 |
| LeftShift | 11 ~ 14 |
| Less | 1 ~ 14 |
| LessEqual | 7 ~ 14 |
| Log | 1 ~ 14 |
| LogSoftmax | 1 ~ 14 |
| LogicalAnd | 1 ~ 14 |
| LogicalNot | 1 ~ 14 |
| LogicalOr | 1 ~ 14 |
| LookupTableFindV2 | 8 ~ 14 |
| LookupTableSizeV2 | 1 ~ 14 |
| Loop | 7 ~ 14 |
| MatMul | 1 ~ 14 |
| MatrixBandPart | 7 ~ 14 |
| MatrixDeterminant | 11 ~ 14 |
| MatrixDiag | 12 ~ 14 |
| MatrixDiagPart | 11 ~ 14 |
| MatrixDiagPartV2 | 11 ~ 14 |
| MatrixDiagPartV3 | 11 ~ 14 |
| MatrixDiagV2 | 12 ~ 14 |
| MatrixDiagV3 | 12 ~ 14 |
| MatrixSetDiagV3 | 12 ~ 14 |
| Max | 1 ~ 14 |
| MaxPool | 1 ~ 14 |
| MaxPool3D | 1 ~ 14 |
| MaxPoolV2 | 1 ~ 14 |
| MaxPoolWithArgmax | 8 ~ 14 |
| Maximum | 1 ~ 14 |
| Mean | 1 ~ 14 |
| Min | 1 ~ 14 |
| Minimum | 1 ~ 14 |
| MirrorPad | 1 ~ 14 |
| Mul | 1 ~ 14 |
| Multinomial | 7 ~ 14 |
| Neg | 1 ~ 14 |
| NoOp | 1 ~ 14 |
| NonMaxSuppressionV2 | 10 ~ 14 |
| NonMaxSuppressionV3 | 10 ~ 14 |
| NonMaxSuppressionV4 | 10 ~ 14 |
| NonMaxSuppressionV5 | 10 ~ 14 |
| NotEqual | 1 ~ 14 |
| OneHot | 1 ~ 14 |
| Pack | 1 ~ 14 |
| Pad | 1 ~ 14 |
| PadV2 | 1 ~ 14 |
| ParallelDynamicStitch | 10 ~ 14 |
| Placeholder | 1 ~ 14 |
| PlaceholderV2 | 1 ~ 14 |
| PlaceholderWithDefault | 1 ~ 14 |
| Pow | 1 ~ 14 |
| Prelu | 1 ~ 14 |
| Prod | 1 ~ 14 |
| QueueDequeueManyV2 | 8 ~ 14 |
| QueueDequeueUpToV2 | 8 ~ 14 |
| QueueDequeueV2 | 8 ~ 14 |
| RFFT | 1 ~ 14 |
| RFFT2D | 1 ~ 14 |
| RaggedGather | 11 ~ 14 |
| RaggedRange | 11 ~ 14 |
| RaggedTensorFromVariant | 13 ~ 14 |
| RaggedTensorToSparse | 11 ~ 14 |
| RaggedTensorToTensor | 11 ~ 14 |
| RaggedTensorToVariant | 13 ~ 14 |
| RandomNormal | 1 ~ 14 |
| RandomNormalLike | 1 ~ 14 |
| RandomShuffle | 10 ~ 14 |
| RandomStandardNormal | 1 ~ 14 |
| RandomUniform | 1 ~ 14 |
| RandomUniformInt | 1 ~ 14 |
| RandomUniformLike | 1 ~ 14 |
| Range | 7 ~ 14 |
| RealDiv | 1 ~ 14 |
| Reciprocal | 1 ~ 14 |
| Relu | 1 ~ 14 |
| Relu6 | 1 ~ 14 |
| Reshape | 1 ~ 14 |
| ResizeBicubic | 7 ~ 14 |
| ResizeBilinear | 7 ~ 14 |
| ResizeNearestNeighbor | 7 ~ 14 |
| ReverseSequence | 8 ~ 14 (Except 9) |
| ReverseV2 | 10 ~ 14 |
| RightShift | 11 ~ 14 |
| Roll | 10 ~ 14 |
| Round | 1 ~ 14 |
| Rsqrt | 1 ~ 14 |
| SampleDistortedBoundingBox | 9 ~ 14 |
| SampleDistortedBoundingBoxV2 | 9 ~ 14 |
| Scan | 7 ~ 14 |
| ScatterNd | 11 ~ 14 |
| SegmentMax | 11 ~ 14 |
| SegmentMean | 11 ~ 14 |
| SegmentMin | 11 ~ 14 |
| SegmentProd | 11 ~ 14 |
| SegmentSum | 11 ~ 14 |
| Select | 7 ~ 14 |
| SelectV2 | 7 ~ 14 |
| Selu | 1 ~ 14 |
| Shape | 1 ~ 14 |
| Sigmoid | 1 ~ 14 |
| Sign | 1 ~ 14 |
| Sin | 7 ~ 14 |
| Sinh | 9 ~ 14 |
| Size | 1 ~ 14 |
| Slice | 1 ~ 14 |
| Softmax | 1 ~ 14 |
| SoftmaxCrossEntropyWithLogits | 7 ~ 14 |
| Softplus | 1 ~ 14 |
| Softsign | 1 ~ 14 |
| SpaceToBatchND | 1 ~ 14 |
| SpaceToDepth | 1 ~ 14 |
| SparseFillEmptyRows | 11 ~ 14 |
| SparseReshape | 11 ~ 14 |
| SparseSegmentMean | 11 ~ 14 |
| SparseSegmentMeanWithNumSegments | 11 ~ 14 |
| SparseSegmentSqrtN | 11 ~ 14 |
| SparseSegmentSqrtNWithNumSegments | 11 ~ 14 |
| SparseSegmentSum | 11 ~ 14 |
| SparseSegmentSumWithNumSegments | 11 ~ 14 |
| SparseSoftmaxCrossEntropyWithLogits | 7 ~ 14 |
| SparseToDense | 11 ~ 14 |
| Split | 1 ~ 14 |
| SplitV | 1 ~ 14 |
| Sqrt | 1 ~ 14 |
| Square | 1 ~ 14 |
| SquaredDifference | 1 ~ 14 |
| SquaredDistance | 12 ~ 14 |
| Squeeze | 1 ~ 14 |
| StatelessIf | 1 ~ 14 |
| StatelessWhile | 7 ~ 14 |
| StopGradient | 1 ~ 14 |
| StridedSlice | 1 ~ 14 |
| StringLower | 10 ~ 14 |
| StringToNumber | 9 ~ 14 |
| StringUpper | 10 ~ 14 |
| Sub | 1 ~ 14 |
| Sum | 1 ~ 14 |
| TFL_CONCATENATION | 1 ~ 14 |
| TFL_DEQUANTIZE | 1 ~ 14 |
| TFL_PRELU | 7 ~ 14 |
| TFL_QUANTIZE | 1 ~ 14 |
| TFL_TFLite_Detection_PostProcess | 11 ~ 14 |
| TFL_WHILE | 7 ~ 14 |
| Tan | 7 ~ 14 |
| Tanh | 1 ~ 14 |
| TensorListFromTensor | 7 ~ 14 |
| TensorListGetItem | 7 ~ 14 |
| TensorListLength | 7 ~ 14 |
| TensorListReserve | 7 ~ 14 |
| TensorListResize | 7 ~ 14 |
| TensorListSetItem | 7 ~ 14 |
| TensorListStack | 7 ~ 14 |
| TensorScatterUpdate | 11 ~ 14 |
| Tile | 1 ~ 14 |
| TopKV2 | 1 ~ 14 |
| Transpose | 1 ~ 14 |
| TruncateDiv | 1 ~ 14 |
| Unique | 11 ~ 14 |
| Unpack | 1 ~ 14 |
| UnsortedSegmentMax | 11 ~ 14 |
| UnsortedSegmentMin | 11 ~ 14 |
| UnsortedSegmentProd | 11 ~ 14 |
| UnsortedSegmentSum | 11 ~ 14 |
| Where | 9 ~ 14 |
| While | 7 ~ 14 |
| ZerosLike | 1 ~ 14 |
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
