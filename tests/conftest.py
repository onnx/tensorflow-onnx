# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" print pytest config."""

from common import get_test_config


def pytest_configure():
    print(get_test_config())
