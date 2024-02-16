# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import sys
from copy import copy
from importlib import import_module
from unittest import mock

import _utils_internal
import pytest

import torch

from _utils_internal import get_default_devices
from torchrl._utils import _rng_decorator, get_binary_env_var, implement_for

from torchrl.envs.libs.gym import gym_backend, GymWrapper, set_gym_backend


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


def uncallable(f):
    class UncallableObject:
        def __init__(self, other):
            for k, v in other.__dict__.items():
                if k not in ("__call__", "__dict__", "__weakref__"):
                    setattr(self, k, v)

    g = UncallableObject(f)
    return g


class implement_for_test_functions:
    """
    Groups functions that are used in tests for `implement_for` decorator.
    """

    @staticmethod
    @implement_for(lambda: import_module("_utils_internal"), "0.3")
    def select_correct_version():
        """To test from+ range and that this function is not selected as the implementation."""
        return "0.3+V1"

    @staticmethod
    @implement_for("_utils_internal", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that this function is selected as the implementation (last implementation)."""
        return "0.3+"

    @staticmethod
    @implement_for(lambda: import_module("_utils_internal"), "0.2", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that right bound is not included."""
        return "0.2-0.3"

    @staticmethod
    @implement_for("_utils_internal", "0.1", "0.2")
    def select_correct_version():  # noqa: F811
        """To test that function with missing from-to range is ignored."""
        return "0.1-0.2"

    @staticmethod
    @implement_for("missing_module")
    def missing_module():
        """To test that calling decorated function with missing module raises an exception."""
        return "missing"

    @staticmethod
    @implement_for("_utils_internal", None, "0.3")
    def missing_version():
        return "0-0.3"

    @staticmethod
    @implement_for("_utils_internal", "0.4")
    def missing_version():  # noqa: F811
        return "0.4+"


def test_implement_for():
    assert implement_for_test_functions.select_correct_version() == "0.3+"


def test_implement_for_missing_module():
    msg = r"Supported version of 'test_utils.implement_for_test_functions.missing_module' has not been found."
    with pytest.raises(ModuleNotFoundError, match=msg):
        implement_for_test_functions.missing_module()


def test_implement_for_missing_version():
    msg = r"Supported version of 'test_utils.implement_for_test_functions.missing_version' has not been found."
    with pytest.raises(ModuleNotFoundError, match=msg):
        implement_for_test_functions.missing_version()


def test_implement_for_reset():
    assert implement_for_test_functions.select_correct_version() == "0.3+"
    _impl = copy(implement_for._implementations)
    name = implement_for.get_func_name(
        implement_for_test_functions.select_correct_version
    )
    for setter in implement_for._setters:
        if implement_for.get_func_name(setter.fn) == name and setter.fn() != "0.3+":
            setter.module_set()
    assert implement_for_test_functions.select_correct_version() != "0.3+"
    implement_for.reset(_impl)
    assert implement_for_test_functions.select_correct_version() == "0.3+"


@pytest.mark.parametrize(
    "version, from_version, to_version, expected_check",
    [
        ("0.21.0", "0.21.0", None, True),
        ("0.21.0", None, "0.21.0", False),
        ("0.9.0", "0.11.0", "0.21.0", False),
        ("0.9.0", "0.1.0", "0.21.0", True),
        ("0.19.99", "0.19.9", "0.21.0", True),
        ("0.19.99", None, "0.19.0", False),
        ("5.61.77", "0.21.0", None, True),
        ("5.61.77", None, "0.21.0", False),
    ],
)
def test_implement_for_check_versions(
    version, from_version, to_version, expected_check
):
    assert (
        implement_for.check_version(version, from_version, to_version) == expected_check
    )


@pytest.mark.parametrize(
    "gymnasium_version, expected_from_version_gymnasium, expected_to_version_gymnasium",
    [
        ("0.27.0", None, None),
        ("0.27.2", None, None),
        ("5.1.77", None, None),
    ],
)
@pytest.mark.parametrize(
    "gym_version, expected_from_version_gym, expected_to_version_gym",
    [
        ("0.21.0", "0.21.0", None),
        ("0.22.0", "0.21.0", None),
        ("5.61.77", "0.21.0", None),
        ("0.9.0", None, "0.21.0"),
        ("0.20.0", None, "0.21.0"),
        ("0.19.99", None, "0.21.0"),
    ],
)
def test_set_gym_environments(
    gym_version,
    expected_from_version_gym,
    expected_to_version_gym,
    gymnasium_version,
    expected_from_version_gymnasium,
    expected_to_version_gymnasium,
):
    # mock gym and gymnasium imports
    mock_gym = uncallable(mock.MagicMock())
    mock_gym.__version__ = gym_version
    mock_gym.__name__ = "gym"
    sys.modules["gym"] = mock_gym

    mock_gymnasium = uncallable(mock.MagicMock())
    mock_gymnasium.__version__ = gymnasium_version
    mock_gymnasium.__name__ = "gymnasium"
    sys.modules["gymnasium"] = mock_gymnasium

    import gym
    import gymnasium

    # look for the right function that should be called according to gym versions (and same for gymnasium)
    for impfor in implement_for._setters:
        if impfor.fn.__name__ == "_set_gym_environments":
            if (impfor.module_name, impfor.from_version, impfor.to_version) == (
                "gym",
                expected_from_version_gym,
                expected_to_version_gym,
            ):
                expected_fn_gym = impfor.fn
            elif (impfor.module_name, impfor.from_version, impfor.to_version) == (
                "gymnasium",
                expected_from_version_gymnasium,
                expected_to_version_gymnasium,
            ):
                expected_fn_gymnasium = impfor.fn

    with set_gym_backend(gymnasium):
        assert (
            _utils_internal._set_gym_environments == expected_fn_gymnasium
        ), expected_fn_gym

    with set_gym_backend(gym):
        assert (
            _utils_internal._set_gym_environments == expected_fn_gym
        ), expected_fn_gymnasium

    with set_gym_backend(gymnasium):
        assert (
            _utils_internal._set_gym_environments == expected_fn_gymnasium
        ), expected_fn_gym


def test_set_gym_environments_no_version_gymnasium_found():
    gymnasium_version = "0.26.0"
    gymnasium_name = "gymnasium"
    mock_gymnasium = uncallable(mock.MagicMock())
    mock_gymnasium.__version__ = gymnasium_version
    mock_gymnasium.__name__ = gymnasium_name
    sys.modules["gymnasium"] = mock_gymnasium

    import gymnasium

    assert gymnasium.__version__ == "0.26.0"

    # this version of gymnasium does not exist in implement_for
    # therefore, set_gym_backend will not set anything and raise an ImportError.
    msg = f"could not set anything related to gym backend {gymnasium_name} with version={gymnasium_version}."
    with pytest.raises(ImportError, match=msg):
        with set_gym_backend(gymnasium):
            _utils_internal._set_gym_environments()


def test_set_gym_backend_types():
    mock_gym = uncallable(mock.MagicMock())
    gym_version = "0.26.0"
    mock_gym.__version__ = gym_version
    mock_gym.__name__ = "gym"
    sys.modules["gym"] = mock_gym

    import gym

    assert not callable(gym)

    with set_gym_backend("gym"):
        assert gym_backend() == gym
    with set_gym_backend(lambda: gym):
        assert gym_backend() == gym
    with set_gym_backend(gym):
        assert gym_backend() == gym


# we check that the order where these funs are defined won't affect which is called
@implement_for("torch", "1.0", "1.8")
def torch_foo():
    return 0


@implement_for("torch", "1.8", None)
def torch_foo():  # noqa: F811
    return 1


@implement_for("torch", None, "1.0")
def torch_foo():  # noqa: F811
    return 1


def test_set_gym_nested():
    mock_gym = uncallable(mock.MagicMock())
    mock_gym.__version__ = "0.21.0"
    mock_gym.__name__ = "gym"
    sys.modules["gym"] = mock_gym

    mock_gymnasium = uncallable(mock.MagicMock())
    mock_gymnasium.__version__ = "0.28.0"
    mock_gymnasium.__name__ = "gymnasium"
    sys.modules["gymnasium"] = mock_gymnasium

    import gym
    import gymnasium

    assert torch_foo() == 1

    class MockGym:
        _is_batched = False

    with set_gym_backend(gym):
        GymWrapper._output_transform(
            MockGym, (1, 2, True, {})
        )  # would break with gymnasium
        assert torch_foo() == 1
        with set_gym_backend(gymnasium):
            GymWrapper._output_transform(
                MockGym, (1, 2, True, True, {})
            )  # would break with gym
            assert torch_foo() == 1
        GymWrapper._output_transform(
            MockGym, (1, 2, True, {})
        )  # would break with gymnasium
    with set_gym_backend("gym"):
        GymWrapper._output_transform(
            MockGym, (1, 2, True, {})
        )  # would break with gymnasium
        assert torch_foo() == 1
        with set_gym_backend("gymnasium"):
            GymWrapper._output_transform(
                MockGym, (1, 2, True, True, {})
            )  # would break with gym
            assert torch_foo() == 1
        GymWrapper._output_transform(
            MockGym, (1, 2, True, {})
        )  # would break with gymnasium


@pytest.mark.parametrize("device", get_default_devices())
def test_rng_decorator(device):
    with torch.device(device):
        torch.manual_seed(10)
        s0a = torch.randn(3)
        with _rng_decorator(0):
            torch.randn(3)
        s0b = torch.randn(3)
        torch.manual_seed(10)
        s1a = torch.randn(3)
        s1b = torch.randn(3)
        torch.testing.assert_close(s0a, s1a)
        torch.testing.assert_close(s0b, s1b)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
