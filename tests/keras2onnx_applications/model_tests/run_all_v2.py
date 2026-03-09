# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

os.environ["PYTHONPATH"] = \
    os.environ.get("PYTHONPATH", "") + os.pathsep + "../../keras2onnx_unit_tests" + os.pathsep + "../../../"
os.environ["TF2ONNX_CATCH_ERRORS"] = "FALSE"

files = ['test_keras_applications_v2.py', 'test_transformers.py', 'test_chatbot.py',
         'test_resnext.py']
files.sort()

res_final = True
for f_ in files:
    res = subprocess.run(
        [sys.executable, "-m", "pytest", f_, "--no-cov",
         "--doctest-modules", f"--junitxml=junit/test-results-{f_[5:-3]}.xml"],
        check=False
    )
    if res.returncode != 0:
        res_final = False

sys.exit(0 if res_final else 1)
