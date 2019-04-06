# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Opset registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import inspect

# pylint: disable=unused-argument,missing-docstring,invalid-name

class tf_op:
    _OPSETS = collections.OrderedDict()
    _MAPPING = None

    def __init__(self, name, domain="onnx", **kwargs):
        if not isinstance(name, list):
            name = [name]
        self.name = name
        self.domain = domain
        self.kwargs = kwargs

    def __call__(self, func):
        opset = tf_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            tf_op._OPSETS[self.domain] = opset
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("version_"):
                version = int(k.replace("version_", ""))
                while version >= len(opset):
                    opset.append({})
                opset_dict = opset[version]
                for name in self.name:
                    opset_dict[name] = (v, self.kwargs)

    @staticmethod
    def get_opsets():
        return tf_op._OPSETS


    @staticmethod
    def create_mapping(max_opset, extra_opsets):
        mapping = {"onnx": max_opset}
        if extra_opsets:
            for extra_opset in extra_opsets:
                mapping[extra_opset.domain] = extra_opset.version
        ops_mapping = {}
        for domain, opsets in tf_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        ops_mapping.update(op_map)

        tf_op._MAPPING = ops_mapping
        return ops_mapping


    @staticmethod
    def find_op(name):
        map_info = tf_op._MAPPING.get(name)
        if map_info is None:
            return None
        return map_info
