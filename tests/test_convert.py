# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" Test convert.py """

import os
import sys
import unittest

from tf2onnx import convert


def run_test_case(args):
    """ run case and clean up """
    sys.argv = args
    convert.main()
    ret = os.path.exists(args[-1])
    if ret:
        os.remove(args[-1])
    return ret


class Tf2OnnxConvertTest(unittest.TestCase):
    """ teat cases for convert.py """

    def test_convert_saved_model(self):
        """ convert saved model """
        self.assertTrue(run_test_case(['',
                                       '--saved-model',
                                       'tests/models/regression/saved_model',
                                       '--output',
                                       'converted_saved_model.onnx']))

    def test_convert_graphdef(self):
        """ convert graphdef """
        self.assertTrue(run_test_case(['',
                                       '--input',
                                       'tests/models/regression/graphdef/frozen.pb',
                                       '--inputs',
                                       'X:0',
                                       '--outputs',
                                       'pred:0',
                                       '--output',
                                       'converted_graphdef.onnx']))

    def test_convert_checkpoint(self):
        """ convert checkpoint """
        self.assertTrue(run_test_case(['',
                                       '--checkpoint',
                                       'tests/models/regression/checkpoint/model.meta',
                                       '--inputs',
                                       'X:0',
                                       '--outputs',
                                       'pred:0',
                                       '--output',
                                       'converted_checkpoint.onnx']))


if __name__ == '__main__':
    unittest.main()
