# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

os.environ["PYTHONPATH"] = \
    os.environ.get("PYTHONPATH", "") + os.pathsep + "../../keras2onnx_unit_tests" + os.pathsep + "../../../"
os.environ["TF2ONNX_CATCH_ERRORS"] = "FALSE"

test_runs = [
    ("test_chatbot.py", "chatbot", False),
    ("test_keras_applications_v2.py", "keras_applications_v2", False),
    # Run each model in a fresh process so TensorFlow releases its native allocations.
    ("test_resnext.py::TestResNext::test_ResNext", "resnext", True),
    ("test_resnext.py::TestResNext::test_SEResNext", "seresnext", True),
    ("test_transformers.py", "transformers", False),
]

res_final = True
for test_target, report_name, stream_output in test_runs:
    pytest_args = [
        sys.executable, "-m", "pytest", test_target, "--no-cov",
        "--doctest-modules", f"--junitxml=junit/test-results-{report_name}.xml",
    ]
    if stream_output:
        pytest_args.append("--capture=no")

    res = subprocess.run(
        pytest_args,
        check=False
    )
    if res.returncode != 0:
        res_final = False

sys.exit(0 if res_final else 1)
