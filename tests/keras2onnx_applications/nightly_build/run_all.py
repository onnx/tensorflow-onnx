# SPDX-License-Identifier: Apache-2.0

import os
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exclude')
args = parser.parse_args()
exclude_set = set(args.exclude.split()) if args.exclude is not None else set()

os.environ["PYTHONPATH"] = \
    os.environ.get("PYTHONPATH", "") + os.pathsep + "../../keras2onnx_unit_tests" + os.pathsep + "../../../"
os.environ["TF2ONNX_CATCH_ERRORS"] = "FALSE"

mypath = '.'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("test_") == 0]
files.sort()

res_final = True
for f_ in files:
    if f_ not in exclude_set:
        res = os.system("pytest " + f_ +  " --no-cov "
                        "--doctest-modules --junitxml=junit/test-results-" + f_[5:-3] + ".xml")
        if res > 0:
            res_final = False

if res_final:
    assert(True)
else:
    assert(False)
