# SPDX-License-Identifier: Apache-2.0


""" print pytest config."""

from common import get_test_config
from tf2onnx import logging


def pytest_configure():
    config = get_test_config()
    logging.basicConfig(level=config.log_level)
    with logging.set_scope_level(logging.INFO) as logger:
        logger.info(config)
