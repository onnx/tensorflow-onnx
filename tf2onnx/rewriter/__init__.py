# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
"""tf2onnx.rewriter module."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tf2onnx.rewriter.cond_rewriter import rewrite_cond
from tf2onnx.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from tf2onnx.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from tf2onnx.rewriter.eye_rewriter import rewrite_eye
from tf2onnx.rewriter.thresholded_relu_rewriter import rewrite_thresholded_relu
from tf2onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm, \
    rewrite_single_direction_gru, rewrite_bi_direction_gru, \
    rewrite_custom_rnn_cell, rewrite_generic_loop

__all__ = [
    "rewrite_cond",
    "rewrite_random_uniform",
    "rewrite_random_uniform_fold_const",
    "rewrite_leakyrelu",
    "rewrite_thresholded_relu",
    "rewrite_eye",
    "rewrite_single_direction_lstm",
    "rewrite_bi_direction_lstm",
    "rewrite_single_direction_gru",
    "rewrite_bi_direction_gru",
    "rewrite_custom_rnn_cell",
    "rewrite_generic_loop"
]
