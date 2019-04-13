# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" print pytest config."""

import logging
from common import get_test_config
from tf2onnx import utils


def pytest_configure():
    config = get_test_config()
    logging.basicConfig(level=logging.WARNING)
    with utils.set_log_level(logging.getLogger(), logging.INFO) as logger:
        logger.info(config)
