# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn - lstm support
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from tf2onnx.rewriter.bilstm_rewriter import rewrite_bidirectional_lstms
from tf2onnx.rewriter.bigru_rewriter import rewrite_bidirectional_grus
from tf2onnx.rewriter.custom_rnn_rewriter import CustomRnnRewriter
from tf2onnx.rewriter.loop_rewriter import LoopRewriter
from tf2onnx.rewriter.lstm_rewriter import LSTMUnitRewriter
from tf2onnx.rewriter.gru_rewriter import GRUUnitRewriter

# pylint: disable=invalid-name,unused-argument,missing-docstring


logger = logging.getLogger(__name__)


def rewrite_single_direction_lstm(g, ops):
    r = LSTMUnitRewriter(g)
    return r.run()


def rewrite_bi_direction_lstm(g, ops):
    return rewrite_bidirectional_lstms(g, ops)


def rewrite_single_direction_gru(g, ops):
    r = GRUUnitRewriter(g)
    return r.run()


def rewrite_bi_direction_gru(g, ops):
    return rewrite_bidirectional_grus(g, ops)


def rewrite_custom_rnn_cell(g, ops):
    return  CustomRnnRewriter(g).run()


def rewrite_generic_loop(g, ops):
    return LoopRewriter(g).run()
