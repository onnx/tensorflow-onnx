# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn - lstm support
"""

from __future__ import division
from __future__ import print_function

from tf2onnx.rewriter.bilstm_rewriter import *
from tf2onnx.rewriter.lstm_rewriter import *
from tf2onnx.rewriter.rnn_utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.rnn")

def rewrite_single_direction_lstm(g, ops):
    r = LSTMUnitRewriter(g)
    return r.run()

def rewrite_bi_direction_lstm(g, ops): 
    return process_bilstm_batch_major(g, ops)