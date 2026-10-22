# SPDX-License-Identifier: Apache-2.0


"""Checks on the dependencies tf2onnx declares in its package metadata."""

import ast
import os
import sys
import unittest
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# pylint: disable=missing-docstring

TF2ONNX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tf2onnx")

# Imported by tf2onnx but intentionally not declared: users install these themselves.
OPTIONAL_MODULES = {"tensorflow", "keras", "tensorflow_text"}
# Imported directly but provided by a declared dependency.
TRANSITIVE_MODULES = {
    "google": "onnx",  # google.protobuf
    "urllib3": "requests",
}


def _declared_requirements():
    try:
        requires = metadata.requires("tf2onnx") or []
    except metadata.PackageNotFoundError as e:
        raise unittest.SkipTest("tf2onnx is not installed (pip install -e .)") from e
    reqs = {}
    for line in requires:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        reqs[canonicalize_name(req.name)] = req
    return reqs


def _imported_top_level_modules():
    modules = {}
    for root, _, files in os.walk(TF2ONNX_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for module in names:
                    top = module.split(".")[0]
                    if top != "tf2onnx" and top not in sys.stdlib_module_names:
                        modules.setdefault(top, path)
    return modules


class PackageMetadataTests(unittest.TestCase):

    def test_onnx_requirement_skips_broken_release(self):
        spec = _declared_requirements()["onnx"].specifier
        self.assertNotIn("1.19.0", spec)
        # The floor and newer releases must stay installable.
        for version in ["1.14.0", "1.16.1", "1.18.0", "1.19.1", "1.23.2"]:
            self.assertIn(version, spec, version)

    def test_imported_modules_are_declared(self):
        declared = _declared_requirements()
        for module, path in sorted(_imported_top_level_modules().items()):
            if module in OPTIONAL_MODULES:
                continue
            dist = canonicalize_name(TRANSITIVE_MODULES.get(module, module))
            self.assertIn(dist, declared, "%s is imported in %s but not declared in pyproject.toml" % (module, path))


if __name__ == "__main__":
    unittest.main()
