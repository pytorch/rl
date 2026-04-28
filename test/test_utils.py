# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import os
import sys
from copy import copy
from importlib import import_module
from unittest import mock

import pytest

import torch

import torchrl.envs.libs.gym as _gym_lib
from packaging import version
from torchrl._utils import _rng_decorator, get_binary_env_var, implement_for
from torchrl.envs.libs.gym import gym_backend, GymWrapper, set_gym_backend

from torchrl.objectives.utils import _pseudo_vmap

from torchrl.testing import get_default_devices, gym_helpers as _gym_helpers

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)


def _clear_gym_implement_for_state():
    """Clear all gym/gymnasium-related state from implement_for caches.

    This is necessary for test isolation when mocking gym/gymnasium modules.
    Clears _cache_modules, _implementations, and resets DEFAULT_GYM, then
    re-resolves the correct implementations based on the real installed backend.
    Does NOT clear _lazy_impl since it contains all registered implementations
    from import time.
    """
    # Reset the global DEFAULT_GYM to force re-initialization
    _gym_lib.DEFAULT_GYM = None

    # Clear module cache to force re-import
    implement_for._cache_modules.pop("gym", None)
    implement_for._cache_modules.pop("gymnasium", None)

    # Clear implementations to force re-evaluation of which implementation wins
    keys_to_remove = [
        k
        for k in implement_for._implementations
        if "gym" in k.lower() or "gymnasium" in k.lower()
    ]
    for k in keys_to_remove:
        implement_for._implementations.pop(k, None)

    # Re-resolve gym-related module-level functions from _lazy_impl.
    # set_gym_backend().module_set() may have replaced module-level functions
    # (like _set_gym_environments, _set_gym_args) with implementations for a
    # mocked backend. We must re-dispatch via _lazy_impl so that implement_for
    # re-evaluates the real installed gym version and calls module_set() with
    # the correct implementation.
    for func_name in list(implement_for._lazy_impl.keys()):
        if "gym" in func_name.lower():
            for local_call in implement_for._lazy_impl[func_name]:
                local_call()


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
    @implement_for(lambda: import_module("torchrl.testing.utils"), "0.3")
    def select_correct_version():
        """To test from+ range and that this function is not selected as the implementation."""
        return "0.3+V1"

    @staticmethod
    @implement_for("torchrl.testing.utils", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that this function is selected as the implementation (last implementation)."""
        return "0.3+"

    @staticmethod
    @implement_for(lambda: import_module("torchrl.testing.utils"), "0.2", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that right bound is not included."""
        return "0.2-0.3"

    @staticmethod
    @implement_for("torchrl.testing.utils", "0.1", "0.2")
    def select_correct_version():  # noqa: F811
        """To test that function with missing from-to range is ignored."""
        return "0.1-0.2"

    @staticmethod
    @implement_for("missing_module")
    def missing_module():
        """To test that calling decorated function with missing module raises an exception."""
        return "missing"

    @staticmethod
    @implement_for("torchrl.testing.utils", None, "0.3")
    def missing_version():
        return "0-0.3"

    @staticmethod
    @implement_for("torchrl.testing.utils", "0.4")
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
        ("0.99.0", "0.21.0", None, True),
        ("0.99.0", None, "0.21.0", False),
    ],
)
def test_implement_for_check_versions(
    version, from_version, to_version, expected_check
):
    assert (
        implement_for.check_version(version, from_version, to_version) == expected_check
    )


@pytest.mark.isolate
@pytest.mark.parametrize(
    "gymnasium_version, expected_from_version_gymnasium, expected_to_version_gymnasium",
    [
        ("0.27.0", None, "1.0.0"),
        ("0.27.2", None, "1.0.0"),
        # ("1.0.1", "1.0.0", None),
    ],
)
@pytest.mark.parametrize(
    "gym_version, expected_from_version_gym, expected_to_version_gym",
    [
        ("0.21.0", "0.21.0", None),
        ("0.22.0", "0.21.0", None),
        ("0.99.0", "0.21.0", None),
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
    # Save original modules to restore after the test
    original_gym = sys.modules.get("gym")
    original_gymnasium = sys.modules.get("gymnasium")

    try:
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
        expected_fn_gymnasium = None
        expected_fn_gym = None
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
                if expected_fn_gym is not None and expected_fn_gymnasium is not None:
                    break

        with set_gym_backend(gymnasium):
            assert (
                _gym_helpers._set_gym_environments is expected_fn_gymnasium
            ), expected_fn_gym

        with set_gym_backend(gym):
            assert (
                _gym_helpers._set_gym_environments is expected_fn_gym
            ), expected_fn_gymnasium

        with set_gym_backend(gymnasium):
            assert (
                _gym_helpers._set_gym_environments is expected_fn_gymnasium
            ), expected_fn_gym

    finally:
        # Restore original modules to avoid polluting other tests
        if original_gym is not None:
            sys.modules["gym"] = original_gym
        else:
            sys.modules.pop("gym", None)
        if original_gymnasium is not None:
            sys.modules["gymnasium"] = original_gymnasium
        else:
            sys.modules.pop("gymnasium", None)
        _clear_gym_implement_for_state()


@pytest.mark.isolate
def test_set_gym_environments_no_version_gymnasium_found():
    # Save original module to restore after the test
    original_gymnasium = sys.modules.get("gymnasium")

    try:
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
                _gym_helpers._set_gym_environments()

    finally:
        # Restore original module to avoid polluting other tests
        if original_gymnasium is not None:
            sys.modules["gymnasium"] = original_gymnasium
        else:
            sys.modules.pop("gymnasium", None)
        _clear_gym_implement_for_state()


@pytest.mark.isolate
def test_set_gym_backend_types():
    # Save original module to restore after the test
    original_gym = sys.modules.get("gym")

    try:
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

    finally:
        # Restore original module to avoid polluting other tests
        if original_gym is not None:
            sys.modules["gym"] = original_gym
        else:
            sys.modules.pop("gym", None)
        _clear_gym_implement_for_state()


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


@pytest.mark.isolate
def test_set_gym_nested():
    # Save original modules to restore after the test
    original_gym = sys.modules.get("gym")
    original_gymnasium = sys.modules.get("gymnasium")

    try:
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

    finally:
        # Restore original modules to avoid polluting other tests
        if original_gym is not None:
            sys.modules["gym"] = original_gym
        else:
            sys.modules.pop("gym", None)
        if original_gymnasium is not None:
            sys.modules["gymnasium"] = original_gymnasium
        else:
            sys.modules.pop("gymnasium", None)
        _clear_gym_implement_for_state()


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


def add_one(x):
    return x + 1


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
)
@pytest.mark.parametrize("in_dim, out_dim", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_vmap_in_out_dims(in_dim, out_dim):
    # Create a tensor with batch dimension
    x = torch.arange(10).reshape(2, 5)
    # Move the input dimension to match in_dim
    x_moved = torch.moveaxis(x, 0, in_dim)
    # Using vmap with specified in_dim and out_dim
    vmapped_add_one = torch.vmap(add_one, in_dims=in_dim, out_dims=out_dim)
    actual_result = vmapped_add_one(x_moved)
    pseudo_vmapped_add_one = _pseudo_vmap(add_one, in_dims=in_dim, out_dims=out_dim)
    pseudo_actual_result = pseudo_vmapped_add_one(x_moved)

    # Expected result by applying add_one on each element of the batch separately
    expected_result = x + 1
    # Move the output dimension to match the expected result
    if out_dim == 1:
        actual_result = torch.moveaxis(actual_result, out_dim, 0)
        pseudo_actual_result = torch.moveaxis(pseudo_actual_result, out_dim, 0)
    # Assert the results are as expected
    assert torch.allclose(actual_result, expected_result)
    assert torch.allclose(pseudo_actual_result, expected_result)


class TestProfilingDecorator:
    """Tests for the TORCHRL_PROFILING-gated profiling decorator."""

    def test_decorator_is_identity_when_unarmed(self, monkeypatch):
        # Force the decorator's import-time gate to "off" regardless of how the
        # test process was launched, so we can assert the zero-overhead branch.
        from torchrl import _utils

        monkeypatch.setattr(_utils, "_PROFILING_ALLOWED", False)

        def fn(x):
            return x + 1

        decorated = _utils._maybe_record_function_decorator("test")(fn)
        assert decorated is fn  # truly identity, no closure

    def test_decorator_wraps_when_armed(self, monkeypatch):
        from torchrl import _utils

        monkeypatch.setattr(_utils, "_PROFILING_ALLOWED", True)
        monkeypatch.setattr(_utils, "_PROFILING_ENABLED", True)

        def fn(x):
            return x + 1

        decorated = _utils._maybe_record_function_decorator("test")(fn)
        assert decorated is not fn
        assert decorated.__wrapped__ is fn  # functools.wraps preserves
        assert decorated(5) == 6

    def test_set_profiling_enabled_warns_when_unarmed(self, monkeypatch):
        from torchrl import _utils

        monkeypatch.setattr(_utils, "_PROFILING_ALLOWED", False)
        monkeypatch.setattr(_utils, "_PROFILING_ENABLED", False)

        with pytest.warns(UserWarning, match="TORCHRL_PROFILING=1 was"):
            _utils.set_profiling_enabled(True)
        # Must remain off — the warning is a no-op gate, not a soft enable.
        assert _utils._PROFILING_ENABLED is False

    def test_set_profiling_enabled_toggles_when_armed(self, monkeypatch):
        from torchrl import _utils

        monkeypatch.setattr(_utils, "_PROFILING_ALLOWED", True)
        monkeypatch.setattr(_utils, "_PROFILING_ENABLED", True)

        _utils.set_profiling_enabled(False)
        assert _utils._PROFILING_ENABLED is False
        _utils.set_profiling_enabled(True)
        assert _utils._PROFILING_ENABLED is True

    def test_maybe_record_function_returns_nullcontext_when_disabled(self, monkeypatch):
        from torchrl import _utils

        monkeypatch.setattr(_utils, "_PROFILING_ENABLED", False)
        ctx = _utils._maybe_record_function("test")
        assert ctx is _utils._NULL_CONTEXT

    def test_as_remote_propagates_torchrl_profiling(self, monkeypatch):
        # Mock the ray module so we can capture what `ray.remote` was called with.
        ray_mock = mock.MagicMock()
        captured = {}

        def fake_remote(**kwargs):
            captured.update(kwargs)
            return lambda cls: cls

        ray_mock.remote = fake_remote
        monkeypatch.setitem(sys.modules, "ray", ray_mock)
        monkeypatch.setenv("TORCHRL_PROFILING", "1")

        from torchrl._utils import as_remote

        class Dummy:
            pass

        as_remote.__func__(Dummy, remote_config={"num_cpus": 1})
        env_vars = captured["runtime_env"]["env_vars"]
        assert env_vars["TORCHRL_PROFILING"] == "1"
        assert captured["num_cpus"] == 1

    def test_as_remote_no_op_when_torchrl_profiling_unset(self, monkeypatch):
        ray_mock = mock.MagicMock()
        captured = {}

        def fake_remote(**kwargs):
            captured.update(kwargs)
            return lambda cls: cls

        ray_mock.remote = fake_remote
        monkeypatch.setitem(sys.modules, "ray", ray_mock)
        monkeypatch.delenv("TORCHRL_PROFILING", raising=False)

        from torchrl._utils import as_remote

        class Dummy:
            pass

        as_remote.__func__(Dummy, remote_config={"num_cpus": 1})
        assert "runtime_env" not in captured


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
