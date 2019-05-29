# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities that match patterns in a tf.Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging

import six


logger = logging.getLogger(__name__)


class OpTypePattern(object):
    """A tree pattern that matches TF expressions with certain op types."""

    def __init__(self, op_type, name=None, inputs=None):
        """Initializes an OpTypePattern.

        Args:
          op_type: string that specifies the allowed types of the root. It can be
            (1) an op type, e.g. 'Conv2D',
            (2) '*', i.e. wildcard, or
            (3) multiple op types separated by '|', e.g., 'Relu|Relu6'.
            We could use regex strings, which might be worthwhile when we have many
            similar TF op types.
          name: Optional string. The name of the pattern that can be looked up in
            MatchResult.
          inputs: Optional list of `OpTypePattern`s or strings that specify the
            patterns for the inputs of a matching op. If None, this pattern accepts
            any inputs of a matching op.
        """
        self._op_type = op_type
        self._name = name
        if inputs is None:
            inputs = []
        self._inputs = [
            input_pattern if isinstance(input_pattern, OpTypePattern) else
            OpTypePattern(input_pattern) for input_pattern in inputs
        ]

    @property
    def op_type(self):
        return self._op_type

    @property
    def inputs(self):
        return self._inputs

    @property
    def name(self):
        return self._name


class MatchResult(object):
    r"""Encapsulates the result of a match done by GraphMatcher.

    MatchResult contains a map from OpTypePattern to the matching op and tensor.
    When the matching op has multiple output tensors, the matching tensor is the
    output tensor used by the matching op of the parent pattern. E.g., when we
    match graph

        -         +
       / \y0   y1/ \
      x    split    z
            |
            y         (nodes are ops; edges are going up)

    against add_pattern defined as

      y1_pattern = OpTypePattern('*')
      z_pattern = OpTypePattern('*')
      add_pattern = OpTypePattern('+', inputs=[y1_pattern, z_pattern])

    the matching op of `y1_pattern` is `split`, and the matching tensor of
    `y1_pattern`
    is `y1` not `y0`.
    """

    def __init__(self):
        self._pattern_to_op_tensor = {}
        self._name_to_pattern = {}

    def add(self, pattern, op, tensor):
        self._pattern_to_op_tensor[pattern] = op, tensor
        if pattern.name is not None:
            # allow this so we can apply subgraphs multiple times
            # if pattern.name in self._name_to_pattern:
            #   raise ValueError(
            #       'Name %s is already bound to another pattern' % pattern.name)
            self._name_to_pattern[pattern.name] = pattern

    def _to_pattern(self, pattern_or_name):
        if isinstance(pattern_or_name, OpTypePattern):
            return pattern_or_name

        if isinstance(pattern_or_name, six.text_type):
            return self._name_to_pattern[pattern_or_name]

        raise ValueError('pattern_or_name has type %s. Expect OpTypePattern or str.'
                         % type(pattern_or_name))

    def get_op(self, pattern_or_name):
        return self._pattern_to_op_tensor[self._to_pattern(pattern_or_name)][0]

    def get_tensor(self, pattern_or_name):
        return self._pattern_to_op_tensor[self._to_pattern(pattern_or_name)][1]

    def get_nodes(self):
        return [n[0] for n in self._pattern_to_op_tensor.values()]


class GraphMatcher(object):
    """Checks if a particular subgraph matches a given pattern."""

    def __init__(self, pattern, allow_reorder=False):
        """Initializes a GraphMatcher.

        Args:
          pattern: The `OpTypePattern` against which `GraphMatcher` matches
            subgraphs.
        """
        self._pattern = pattern
        self._allow_reorder = allow_reorder

    def _match_pattern(self, pattern, op, tensor):
        """Returns whether an TF expression rooted at `op` matches `pattern`.

        If there is a match, adds to `self._match_result` the matching op and tensor
        with key `pattern`.

        Args:
          pattern: An `OpTypePattern`.
          op: A `tf.Operation` to match against the pattern.
          tensor: the output `tf.Tensor` of `op` that is used by the matching op of
            `pattern`'s parent. Can be None if `pattern` is already the root of the
            pattern tree.

        Returns:
          True if an TF expression rooted at `op` matches `pattern`.
        """
        if pattern.op_type is None:
            return True

        if pattern.op_type != '*':
            if op is None or op.type not in pattern.op_type.split('|'):
                logger.debug(
                    "mismatched type at %s: [%s, %s]",
                    op.name if op else "None",
                    pattern.op_type,
                    op.type if op else "None"
                )
                return False

        self._match_result.add(pattern, op, tensor)
        # print("matched", ",".join([op.type + "|" + op.name for op in self._match_result.get_nodes()]))

        if not pattern.inputs:
            # If pattern.inputs is empty, skips the rest and accepts all the inputs.
            return True

        if not op or len(op.inputs) != len(pattern.inputs):
            logger.debug(
                "mismatched input number at %s: [%s, %s]",
                op.name if op else "None",
                len(pattern.inputs),
                len(op.inputs)
            )
            return False

        if self._allow_reorder:
            inputs = [None] * len(op.inputs)
            wanted = copy.copy(pattern.inputs)
            for idx, i in enumerate(op.inputs):
                for j in range(len(wanted)):  # pylint: disable=consider-using-enumerate
                    if i.type == wanted[j].op_type:
                        inputs[idx] = wanted[j]
                        del wanted[j]
                        break
            for idx, i in enumerate(inputs):
                if i is None:
                    inputs[idx] = wanted[0]
                    del wanted[0]
            pat = list(zip(op.inputs, inputs))
        else:
            pat = list(zip(op.inputs, pattern.inputs))

        ret = []
        for input_tensor, input_pattern in pat:
            # print("MATCHING", input_pattern.op_type, input_tensor.type)
            r = self._match_pattern(input_pattern, input_tensor, input_tensor)
            ret.append(r)
        return all(ret)

    def match_op(self, op):
        """Matches `op` against `self._pattern`.

        Args:
          op: `tf.Operation` to match against the pattern.

        Returns:
          Returns a `MatchResult` if `op` matches the pattern; otherwise, returns
          None.
        """
        logger.debug("match %s against the pattern", op.name)
        self._match_result = MatchResult()
        if not self._match_pattern(self._pattern, op, tensor=None):
            return None
        return self._match_result

    def match_ops(self, ops):
        """Matches each operation in `ops` against `self._pattern`.

        Args:
          ops: collection of `tf.Operation` to match against the pattern.

        Yields:
          `MatchResult` for each `tf.Operation` that matches the pattern.
        """
        for op in ops:
            match_result = self.match_op(op)
            if match_result:
                yield match_result

    def match_graph(self, graph):
        """Matches each operation in `graph` against `self._pattern`.

        Args:
          graph: `tf.Graph` containing operations to match.

        Yields:
          `MatchResult` for each `tf.Operation` in `graph` that matches the pattern.
        """
        # Python 3.3.2+ implements `yield from`, but for now:
        for match_result in self.match_ops(graph.get_operations()):
            yield match_result
