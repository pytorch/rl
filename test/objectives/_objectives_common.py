# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib.util
import sys
from copy import deepcopy

import pytest
import torch
from packaging import version
from tensordict import assert_allclose_td, TensorDictBase
from tensordict._C import unravel_keys
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    set_composite_lp_aggregate,
)
from torch import nn

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase

_has_functorch = True
try:
    import functorch as ft  # noqa

    make_functional_with_buffers = ft.make_functional_with_buffers
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)
    make_functional_with_buffers = None

_has_transformers = bool(importlib.util.find_spec("transformers"))
_has_botorch = bool(importlib.util.find_spec("botorch"))

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"


class _check_td_steady:
    def __init__(self, td):
        self.td_clone = td.clone()
        self.td = td

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert_allclose_td(
            self.td,
            self.td_clone,
            intersection=True,
            msg="Some keys have been modified in the tensordict!",
        )


def get_devices():
    devices = [torch.device("cpu")]
    for i in range(torch.cuda.device_count()):
        devices += [torch.device(f"cuda:{i}")]
    return devices


class MARLEnv(EnvBase):
    def __init__(self):
        batch = self.batch = (3,)
        super().__init__(batch_size=batch)
        self.n_agents = n_agents = (4,)
        self.obs_feat = obs_feat = (5,)

        self.full_observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(batch + n_agents + obs_feat),
                shape=batch + n_agents,
            ),
            shape=batch,
        )
        self.full_done_spec = Composite(
            done=Unbounded(batch + (1,), dtype=torch.bool),
            terminated=Unbounded(batch + (1,), dtype=torch.bool),
            truncated=Unbounded(batch + (1,), dtype=torch.bool),
            shape=batch,
        )

        self.act_feat_dirich = act_feat_dirich = (10, 2)
        self.act_feat_categ = act_feat_categ = (7,)
        self.full_action_spec = Composite(
            agents=Composite(
                dirich=Unbounded(batch + n_agents + act_feat_dirich),
                categ=Unbounded(batch + n_agents + act_feat_categ),
                shape=batch + n_agents,
            ),
            shape=batch,
        )

        self.full_reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(batch + n_agents + (1,)), shape=batch + n_agents
            ),
            shape=batch,
        )

    @classmethod
    def make_composite_dist(cls):
        dist_cstr = functools.partial(
            CompositeDistribution,
            distribution_map={
                (
                    "agents",
                    "dirich",
                ): lambda concentration: torch.distributions.Independent(
                    torch.distributions.Dirichlet(concentration), 1
                ),
                ("agents", "categ"): torch.distributions.Categorical,
            },
        )
        return ProbabilisticTensorDictModule(
            in_keys=["params"],
            out_keys=[("agents", "dirich"), ("agents", "categ")],
            distribution_class=dist_cstr,
            return_log_prob=True,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        ...

    def _reset(self, tensordic):
        ...

    def _set_seed(self, seed: int | None) -> None:
        ...


class LossModuleTestBase:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert hasattr(
            cls, "test_reset_parameters_recursive"
        ), "Please add a test_reset_parameters_recursive test for this class"

    def _flatten_in_keys(self, in_keys):
        return [
            in_key if isinstance(in_key, str) else "_".join(list(unravel_keys(in_key)))
            for in_key in in_keys
        ]

    def tensordict_keys_test(self, loss_fn, default_keys, td_est=None):
        self.tensordict_keys_unknown_key_test(loss_fn)
        self.tensordict_keys_default_values_test(loss_fn, default_keys)
        self.tensordict_set_keys_test(loss_fn, default_keys)

    def tensordict_keys_unknown_key_test(self, loss_fn):
        """Test that exception is raised if an unknown key is set via .set_keys()"""
        test_fn = deepcopy(loss_fn)

        with pytest.raises(ValueError):
            test_fn.set_keys(unknown_key="test2")

    def tensordict_keys_default_values_test(self, loss_fn, default_keys):
        test_fn = deepcopy(loss_fn)

        for key, value in default_keys.items():
            assert getattr(test_fn.tensor_keys, key) == value

    def tensordict_set_keys_test(self, loss_fn, default_keys):
        """Test setting of tensordict keys via .set_keys()"""
        test_fn = deepcopy(loss_fn)

        new_key = "test1"
        for key, _ in default_keys.items():
            test_fn.set_keys(**{key: new_key})
            assert getattr(test_fn.tensor_keys, key) == new_key

        test_fn = deepcopy(loss_fn)
        test_fn.set_keys(**{key: new_key for key, _ in default_keys.items()})

        for key, _ in default_keys.items():
            assert getattr(test_fn.tensor_keys, key) == new_key

    def set_advantage_keys_through_loss_test(
        self, loss_fn, td_est, loss_advantage_key_mapping
    ):
        key_mapping = loss_advantage_key_mapping
        test_fn = deepcopy(loss_fn)

        new_keys = {}
        for loss_key, (_, new_key) in key_mapping.items():
            new_keys[loss_key] = new_key

        test_fn.set_keys(**new_keys)
        test_fn.make_value_estimator(td_est)

        for _, (advantage_key, new_key) in key_mapping.items():
            assert (
                getattr(test_fn.value_estimator.tensor_keys, advantage_key) == new_key
            )

    @classmethod
    def reset_parameters_recursive_test(cls, loss_fn):
        def get_params(loss_fn):
            for key, item in loss_fn.__dict__.items():
                if isinstance(item, nn.Module):
                    module_name = key
                    params_name = f"{module_name}_params"
                    target_name = f"target_{module_name}_params"
                    params = loss_fn._modules.get(params_name, None)
                    target = loss_fn._modules.get(target_name, None)

                    if params is not None:
                        yield params_name, params._param_td

                    else:
                        for subparam_name, subparam in loss_fn.named_parameters():
                            if module_name in subparam_name:
                                yield subparam_name, subparam

                    if target is not None:
                        yield target_name, target

        old_params = {}

        for param_name, param in get_params(loss_fn):
            with torch.no_grad():
                # Change the parameter to ensure that reset will change it again
                param += 1000
            old_params[param_name] = param.clone()

        loss_fn.reset_parameters_recursive()

        for param_name, param in get_params(loss_fn):
            old_param = old_params[param_name]
            assert (param != old_param).any()
