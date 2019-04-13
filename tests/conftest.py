# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" print pytest config."""

import logging
from common import get_test_config
from tf2onnx import constants, utils


def pytest_configure():
    config = get_test_config()
    logging.basicConfig(level=logging.WARNING, format=constants.LOG_FORMAT)
    with utils.set_log_level(logging.getLogger(), logging.INFO) as logger:
        logger.info(config)
