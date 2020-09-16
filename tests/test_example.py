# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Test examples."""

import os
import subprocess
import unittest
from common import check_opset_min_version, check_opset_max_version, check_tf_min_version


class TestExample(unittest.TestCase):
    """test examples"""

    def run_example(self, name, expected=None):
        "Executes one example."
        full = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "..", "examples", name)
        if not os.path.exists(full):
            raise FileNotFoundError(full)
        proc = subprocess.run(('python %s' % full).split(),
                              capture_output=True, check=True)
        self.assertEqual(0, proc.returncode)
        if expected is not None:
            out = proc.stdout.decode('ascii')
            for exp in expected:
                self.assertIn(exp, out)
        err = proc.stderr.decode('ascii')
        self.assertTrue(err is not None)

    @check_tf_min_version("2.0", "use tf.keras")
    @check_opset_min_version(12)
    @check_opset_max_version(13)
    def test_end2end_tfkeras(self):
        self.run_example(
            "end2end_tfkeras.py",
            expected=["ONNX model is saved at simple_rnn.onnx",
                      "Optimizing ONNX model",
                      "Using opset <onnx, 12>"])


if __name__ == '__main__':
    unittest.main()
