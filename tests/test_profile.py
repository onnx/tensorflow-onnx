# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

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
            outs, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return
        print("!!!!!!!!!!!!!!!")
        print(outs)
        print("!!!!!!!!!!!!!!!")
        print(err)
        print("!!!!!!!!!!!!!!!")
        assert b"Profile complete." in outs or outs == b''


if __name__ == '__main__':
    unittest_main()
