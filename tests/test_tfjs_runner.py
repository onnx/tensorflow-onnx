# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for tfjs_runer.py"""

import json
import unittest

import numpy as np

from tfjs_runner import numpy_to_json, json_to_numpy

# pylint: disable=missing-docstring


class TestTfjsRunner(unittest.TestCase):
    def test_tfjs_runner(self):
        float_array = np.array([[1.1, 2.2], [3.3, 4.4]], np.float32)
        int_array = np.array([[1, 2], [3, 4]], np.int32)
        bool_array = np.array([[True, False], [True, True]], bool)
        string_array = np.array([['Hello world', ''], ['π', 'Tensor']], np.str)
        complex_array = np.array([[1 + 0.1j, 2 + 0.2j], [3 + 0.3j, 4 + 0.4j]], np.complex64)

        arrays = [float_array, int_array, bool_array, string_array, complex_array]
        for arr in arrays:
            array_enc = json.dumps(numpy_to_json(arr))
            print(array_enc)
            array_dec = json_to_numpy(json.loads(array_enc))
            np.testing.assert_equal(arr, array_dec)


if __name__ == '__main__':
    unittest.main()
