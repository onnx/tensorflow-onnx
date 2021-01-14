# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" Test convert.py """

import os
import sys
import unittest

from tf2onnx import convert
from common import check_tf_min_version

def run_test_case(args, paths_to_check=None):
    """ run case and clean up """
    if paths_to_check is None:
        paths_to_check = [args[-1]]
    sys.argv = args
    convert.main()
    ret = True
    for p in paths_to_check:
        if os.path.exists(p):
            os.remove(p)
        else:
            ret = False
    return ret


class Tf2OnnxConvertTest(unittest.TestCase):
    """ teat cases for convert.py """

    def test_convert_saved_model(self):
        """ convert saved model """
        self.assertTrue(run_test_case(['',
                                       '--saved-model',
                                       'tests/models/regression/saved_model',
                                       '--tag',
                                       'serve',
                                       '--output',
                                       'converted_saved_model.onnx']))

    def test_convert_output_frozen_graph(self):
        """ convert saved model """
        self.assertTrue(run_test_case(['',
                                       '--saved-model',
                                       'tests/models/regression/saved_model',
                                       '--tag',
                                       'serve',
                                       '--output',
                                       'converted_saved_model.onnx',
                                       '--output_frozen_graph',
                                       'frozen_graph.pb'
                                      ],
                                      paths_to_check=['converted_saved_model.onnx', 'frozen_graph.pb']))

    @check_tf_min_version("2.2")
    def test_convert_large_model(self):
        """ convert saved model to onnx large model format """
        self.assertTrue(run_test_case(['',
                                       '--large_model',
                                       '--saved-model',
                                       'tests/models/regression/saved_model',
                                       '--tag',
                                       'serve',
                                       '--output',
                                       'converted_saved_model.zip']))

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
