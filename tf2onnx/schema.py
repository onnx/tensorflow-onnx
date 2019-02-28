# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.schema
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
from onnx import defs

def version_mapping(schemas):
    """Obtain version to schema mapping."""
    schemas_with_version = defaultdict(dict)
    for s in schemas:
        schemas_with_version[s.name][int(s.since_version)] = s
    for name, s in schemas_with_version.items():
        schemas_with_version[name] = OrderedDict(
            sorted(schemas_with_version[name].items(), key=lambda x: x[0])
        )
    return schemas_with_version


class ONNXSchema:
    """Wrapper for ONNX schema"""

    all_schemas = defs.get_all_schemas_with_history()
    schemas_with_version = version_mapping(all_schemas)

    @staticmethod
    def get_schema(name, version):
        """Get schema by name within specific version."""
        if name not in ONNXSchema.schemas_with_version:
            return None
        if version < 1:
            return None
        schemas = ONNXSchema.schemas_with_version[name]
        versions = list(schemas.keys())
        for i, v in enumerate(versions):
            if version < v:
                return schemas[versions[i-1]]
        return schemas[versions[-1]]

    @staticmethod
    def get_attribute(name, version):
        """Get valid attributes by op's name and specific version"""
        schema = ONNXSchema.get_schema(name, version)
        if not schema:
            return {}
        return schema.attributes
