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


ONNX_DOMAIN = ""


class OnnxOpSchema(object):
    """Wrapper for Onnx schema."""

    def __init__(self, name, domain, since_version, attributes):
        """Create a Onnx schema
        Args:
            name (str): op name
            attributes (List[str]): valid attributes
            domain (str): default value "" means it's Onnx domain
            since_version (int): opset version, default is 1
        """
        self._name = name
        self._domain = domain
        self._attributes = attributes
        self._since_version = since_version

    @property
    def attributes(self):
        return self._attributes

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name

    @property
    def since_version(self):
        return self._since_version

    @staticmethod
    def from_onnx_schema(onnx_schema):
        name = onnx_schema.name
        domain = onnx_schema.domain
        since_version = int(onnx_schema.since_version)
        attributes = onnx_schema.attributes
        return OnnxOpSchema(name, domain, since_version, attributes)

    def has_attribute(self, attr):
        return attr in self.attributes


def _register_all_schemas_with_history():
    """Register all schemas with history"""
    onnx_schemas = defs.get_all_schemas_with_history()
    name_domain_version_schema_map = defaultdict(lambda: defaultdict(dict))
    for s in onnx_schemas:
        schema = OnnxOpSchema.from_onnx_schema(s)
        name_domain_version_schema_map[schema.name][schema.domain][schema.since_version] = schema

    ordered_map = defaultdict(lambda: defaultdict(OrderedDict))
    for name, domain_version_schema_map in name_domain_version_schema_map.items():
        for domain, version_schema_map in domain_version_schema_map.items():
            ordered_map[name][domain] = OrderedDict(
                sorted(version_schema_map.items(), key=lambda x: -x[0])
            )
    return ordered_map


# format is <OpName, <Domain, <SinceVersion, OpSchema>>>
# SinceVersion is sorted from high to low
_schemas = _register_all_schemas_with_history()


def get_schema(name, max_inclusive_opset_version, domain=ONNX_DOMAIN):
    """Get schema by name within specific version."""
    domain_version_schema_map = _schemas[name]
    version_schema_map = domain_version_schema_map[domain]
    for version, schema in version_schema_map.items():
        if version <= max_inclusive_opset_version:
            return schema
    return None
