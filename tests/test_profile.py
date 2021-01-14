# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for Benchmarks."""
import os
import subprocess
from backend_test_base import Tf2OnnxBackendTestBase
from common import (
    check_opset_min_version, check_tf_min_version,
    unittest_main, check_onnxruntime_min_version
)

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,cell-var-from-loop
# pylint: disable=invalid-name
# pylint: enable=invalid-name

class ProfileTests(Tf2OnnxBackendTestBase):

    folder = os.path.join(os.path.dirname(__file__), '..', 'tools')

    @check_tf_min_version("2.0")
    @check_opset_min_version(12)
    @check_onnxruntime_min_version('1.4.0')
    def test_profile_conversion_time(self):
        filename = os.path.join(ProfileTests.folder, 'profile_conversion_time.py')
        proc = subprocess.Popen(
            ["python", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs = proc.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            return
        assert b"Profile complete." in outs or outs == b''


if __name__ == '__main__':
    unittest_main()
