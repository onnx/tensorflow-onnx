# SPDX-License-Identifier: Apache-2.0


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
        proc = subprocess.run(['python', full],
                              capture_output=True, check=True)
        self.assertEqual(0, proc.returncode)
        out = proc.stdout.decode('ascii')
        if 'tensorflow_hub not installed' in out:
            return
        err = proc.stderr.decode('ascii')
        self.assertTrue(err is not None)
        if expected is not None:
            for exp in expected:
                self.assertIn(exp, out)

    @check_tf_min_version("2.3", "use tf.keras")
    @check_opset_min_version(12)
    @check_opset_max_version(13)
    def test_end2end_tfkeras(self):
        self.run_example(
            "end2end_tfkeras.py",
            expected=["ONNX model is saved at simple_rnn.onnx",
                      "Optimizing ONNX model",
                      "Using opset <onnx, 12>"])

    @check_tf_min_version("2.3", "use tf.keras")
    @check_opset_min_version(12)
    @check_opset_max_version(13)
    def test_end2end_tfhub(self):
        self.run_example(
            "end2end_tfhub.py",
            expected=["ONNX model is saved at efficientnetb0clas.onnx",
                      "Optimizing ONNX model",
                      "Using opset <onnx, 12>"])

    @check_tf_min_version("2.3", "use tf.keras")
    @check_opset_min_version(13)
    @check_opset_max_version(13)
    def test_getting_started(self):
        self.run_example(
            "getting_started.py",
            expected=["Conversion succeeded"])


if __name__ == '__main__':
    unittest.main()
