# SPDX-License-Identifier: Apache-2.0
"""
Logging utilities with custom VERBOSE level and TensorFlow 2 support.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

import tensorflow as tf

from . import constants


# ----------------------------------------------------------------------
# Custom VERBOSE level
# ----------------------------------------------------------------------

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


class Logger(logging.Logger):
    """Custom logger supporting .verbose()."""

    def verbose(self, msg, *args, **kwargs):
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, msg, args, **kwargs)


logging.setLoggerClass(Logger)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

_BASIC_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_VERBOSE_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"


def basic_config(level: int = logging.INFO, **kwargs) -> None:
    """Configure root logger and TensorFlow verbosity."""

    if "format" not in kwargs:
        kwargs["format"] = (
            _BASIC_FORMAT if level >= logging.INFO else _VERBOSE_FORMAT
        )

    logging.basicConfig(level=level, **kwargs)
    set_tf_verbosity(level)


# ----------------------------------------------------------------------
# Verbosity utilities
# ----------------------------------------------------------------------

_LOG_LEVELS = [
    logging.FATAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    VERBOSE,
    logging.DEBUG,
]


def get_verbosity_level(
    verbosity: Optional[int],
    base_level: int = logging.INFO,
) -> int:
    """Convert verbosity offset to logging level."""
    if verbosity is None:
        return base_level

    idx = _LOG_LEVELS.index(base_level)
    idx = min(max(0, verbosity + idx), len(_LOG_LEVELS) - 1)
    return _LOG_LEVELS[idx]


def set_level(level: int) -> None:
    """Set logging level for this package and TF."""
    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)
    logger.setLevel(level)
    set_tf_verbosity(level)


# ----------------------------------------------------------------------
# TensorFlow 2 logging control
# ----------------------------------------------------------------------

def set_tf_verbosity(level: int) -> None:
    """Control TensorFlow logging verbosity (TF ≥ 2.x)."""

    # TensorFlow Python logging
    tf.get_logger().setLevel(level)

    # C++ backend logging
    if level <= logging.INFO:
        tf_cpp_min_log_level = "0"
    elif level <= logging.WARNING:
        tf_cpp_min_log_level = "1"
    elif level <= logging.ERROR:
        tf_cpp_min_log_level = "2"
    else:
        tf_cpp_min_log_level = "3"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_cpp_min_log_level


# ----------------------------------------------------------------------
# Context manager
# ----------------------------------------------------------------------

@contextmanager
def set_scope_level(
    level: int,
    logger: Optional[logging.Logger] = None,
) -> Iterator[logging.Logger]:
    """Temporarily change logger level."""
    logger = logger or logging.getLogger()

    previous = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(previous)
