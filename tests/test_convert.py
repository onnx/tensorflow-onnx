import sys
import unittest
import os
from tf2onnx import convert

def run_test_case(args):
    sys.argv = args
    convert.main()
    ret = os.path.exists(args[-1])
    if ret == True:
        os.remove(args[-1])
    return ret

class Tf2OnnxConvertTest(unittest.TestCase):
    def test_convert_saved_model(self):
        self.assertTrue(run_test_case(['', '--saved-model', 'tests/models/regression/saved_model', '--output', 'converted_saved_model.onnx']))
    def test_convert_graphdef(self):
        self.assertTrue(run_test_case(['', '--input', 'tests/models/regression/graphdef/frozen.pb', '--inputs', 'X:0', '--outputs', 'pred:0', '--output', 'converted_graphdef.onnx']))
    def test_convert_checkpoint(self):
        self.assertTrue(run_test_case(['', '--checkpoint', 'tests/models/regression/checkpoint/model.meta', '--inputs', 'X:0', '--outputs', 'pred:0', '--output', 'converted_checkpoint.onnx']))

if __name__ == '__main__':
    unittest.main()