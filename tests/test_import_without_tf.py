# SPDX-License-Identifier: Apache-2.0


"""`import tf2onnx` and the ONNX-side graph API must work without TensorFlow installed (#2532)."""

import os
import subprocess
import sys
import unittest

# pylint: disable=missing-docstring

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs in a fresh interpreter in which any `import tensorflow` raises ModuleNotFoundError,
# as it would if TensorFlow were not installed.
_SCRIPT = r'''
import importlib.abc
import sys


class _BlockTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by the test)", name=name)
        return None


sys.meta_path.insert(0, _BlockTensorFlow())

import tf2onnx
from onnx import TensorProto, helper
from tf2onnx import logging
from tf2onnx.graph import GraphUtil

logging.set_level(logging.INFO)  # also reaches set_tf_verbosity()

# The ONNX-side optimizer must run too: two chained Identity nodes collapse to one.
graph = helper.make_graph(
    [helper.make_node("Identity", ["x"], ["mid"], name="identity_1"),
     helper.make_node("Identity", ["mid"], ["y"], name="identity_2")],
    "identity_chain",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
optimized = GraphUtil.optimize_model_proto(model, catch_errors=False)
counts = GraphUtil.get_node_count_from_onnx_graph(optimized.graph)
assert counts.get("Identity") == 1, counts

loaded = sorted(m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow."))
assert not loaded, loaded
print("ok")
'''


class TestImportWithoutTensorFlow(unittest.TestCase):
    def test_import_without_tensorflow(self):
        result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
