"""Pytest configuration for ex2ve test harness.

CLI options:
    --exodiff PATH   Path to the exodiff binary (or pyexodiff.py) used by the
                     ``exodiff``-type tests. Default: 'exodiff' on $PATH.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--exodiff",
        action="store",
        default="exodiff",
        help="Path to the exodiff utility used by exodiff-type tests "
             "(default: 'exodiff' on PATH).",
    )


@pytest.fixture
def exodiff(request):
    return request.config.getoption("--exodiff")
