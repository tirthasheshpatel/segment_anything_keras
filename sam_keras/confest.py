# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Tirth Patel.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run_slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
