# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" print pytest config."""

from common import get_test_config
from tf2onnx import logging


def pytest_configure():
    config = get_test_config()
    logging.basicConfig(level=config.log_level)
    with logging.set_scope_level(logging.INFO) as logger:
        logger.info(config)
