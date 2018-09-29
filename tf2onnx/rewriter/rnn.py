# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.rnn - lstm support
"""

from __future__ import division
from __future__ import print_function

import collections
import logging
import numpy as np
import tf2onnx
from onnx import helper, defs, numpy_helper, checker, onnx_pb
from onnx import AttributeProto, TensorProto, GraphProto
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import *
from tf2onnx.rewriter.lstm_rewriter import *
from tf2onnx.rewriter.rnn_utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.rnn")

def rewrite_single_direction_lstm(g, ops):
    r = LSTMUnitRewriter(g)
    return r.run()