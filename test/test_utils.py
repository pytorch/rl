# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
from torchrl._utils import get_binary_env_var, implement_for


@pytest.mark.parametrize("value", ["True", "1", "true"])
def test_get_binary_env_var_positive(value):
    try:
        key = "SOME_ENVIRONMENT_VARIABLE_UNLIKELY_TO_BE_IN_ENVIRONMENT"

        assert key not in os.environ

        os.environ[key] = value
        assert get_binary_env_var(key)

    finally:
        if key in os.environ:
            del os.environ[key]


@pytest.mark.parametrize("value", ["False", "0", "false"])
def test_get_binary_env_var_negative(value):
    try:
        key = "SOME_ENVIRONMENT_VARIABLE_UNLIKELY_TO_BE_IN_ENVIRONMENT"

        assert key not in os.environ

        os.environ[key] = "True"
        assert get_binary_env_var(key)
        os.environ[key] = value
        assert not get_binary_env_var(key)

    finally:
        if key in os.environ:
            del os.environ[key]


def test_get_binary_env_var_missing():
    try:
        key = "SOME_ENVIRONMENT_VARIABLE_UNLIKELY_TO_BE_IN_ENVIRONMENT"

        assert key not in os.environ
        assert not get_binary_env_var(key)

    finally:
        if key in os.environ:
            del os.environ[key]


def test_get_binary_env_var_wrong_value():
    try:
        key = "SOME_ENVIRONMENT_VARIABLE_UNLIKELY_TO_BE_IN_ENVIRONMENT"

        assert key not in os.environ
        os.environ[key] = "smthwrong"
        with pytest.raises(ValueError):
            get_binary_env_var(key)

    finally:
        if key in os.environ:
            del os.environ[key]


# To test from+ range and that this function is correctly selected as the implementation.
@implement_for("_utils_internal", "0.3")
def implement_for_test_func():
    return "0.3+"


# To test that right bound is not included.
@implement_for("_utils_internal", "0.2", "0.3")
def implement_for_test_func():  # noqa: F811
    return "0.2-0.3"


# To test that function with missing from-to range is ignored.
@implement_for("_utils_internal", "0.1", "0.2")
def implement_for_test_func():  # noqa: F811
    return "0.1-0.2"


# To test that incorrect/missing module doesn't raise an import time exception.
@implement_for("missing_module", "0")
def implement_for_test_missing_module():
    return "missing"


@implement_for("_utils_internal", None, "0.3")
def implement_for_test_missing_version():
    return "0-0.3"


@implement_for("_utils_internal", "0.4")
def implement_for_test_missing_version():  # noqa: F811
    return "0.4+"


def test_implement_for():
    assert implement_for_test_func() == "0.3+"


def test_implement_for_missing_module():
    assert implement_for_test_missing_module() == "missing"


def test_implement_for_missing_version():
    assert implement_for_test_missing_version() == "0.4+"
