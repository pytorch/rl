# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
from time import sleep
from omegaconf import open_dict

import pytest
import torch
from _utils_internal import generate_seeds, get_available_devices
from torchrl._utils import timeit

try:
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore

    _has_hydra = True
except ImportError:
    _has_hydra = False
from mocking_classes import (
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
    DiscreteActionVecMockEnv,
)
from packaging import version
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms.transforms import _has_tv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules.tensordict_module.common import _has_functorch
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.helpers.envs import EnvConfig
from torchrl.trainers.helpers.models import (
    DDPGModelConfig,
    DiscreteModelConfig,
    make_ddpg_actor,
    make_dqn_actor,
    make_a2c_model,
    make_ppo_model,
    make_redq_model,
    make_sac_model,
    A2CModelConfig,
    PPOModelConfig,
    REDQModelConfig,
    SACModelConfig,
    DreamerConfig,
    make_dreamer,
)
from torchrl.trainers.helpers.losses import (
    make_a2c_loss,
    A2CLossConfig,
)

TORCH_VERSION = version.parse(torch.__version__)
if TORCH_VERSION < version.parse("1.12.0"):
    UNSQUEEZE_SINGLETON = True
else:
    UNSQUEEZE_SINGLETON = False

## these tests aren't truly unitary but setting up a fake env for the
# purpose of building a model with args is a lot of unstable scaffoldings
# with unclear benefits


@pytest.fixture
def dreamer_constructor_fixture():
    import os

    # we hack the env constructor
    import sys

    sys.path.append(os.path.dirname(__file__) + "/../examples/dreamer/")
    from dreamer_utils import transformed_env_constructor

    yield transformed_env_constructor
    sys.path.pop()


def _assert_keys_match(td, expeceted_keys):
    td_keys = list(td.keys())
    d = set(td_keys) - set(expeceted_keys)
    assert len(d) == 0, f"{d} is in tensordict but unexpected: {td.keys()}"
    d = set(expeceted_keys) - set(td_keys)
    assert len(d) == 0, f"{d} is expected but not in tensordict: {td.keys()}"
    assert len(td_keys) == len(expeceted_keys)


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.skipif(not _has_tv, reason="No torchvision library found")
@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("noisy", [tuple(), ("noisy=True",)])
@pytest.mark.parametrize("distributional", [tuple(), ("distributional=True",)])
@pytest.mark.parametrize("from_pixels", [tuple(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize(
    "categorical_action_encoding",
    [("categorical_action_encoding=True",), ("categorical_action_encoding=False",)],
)
def test_dqn_maker(
    device, noisy, distributional, from_pixels, categorical_action_encoding
):
    flags = list(noisy + distributional + from_pixels + categorical_action_encoding) + [
        "env_name=CartPole-v1"
    ]

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DiscreteModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        env_maker = (
            DiscreteActionConvMockEnvNumpy if from_pixels else DiscreteActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker(
            categorical_action_encoding=cfg.categorical_action_encoding
        )

        actor = make_dqn_actor(proof_environment, cfg, device)
        td = proof_environment.reset().to(device)
        if UNSQUEEZE_SINGLETON and not td.ndimension():
            # Linear and conv used to break for non-batched data
            actor(td.unsqueeze(0))
        else:
            actor(td)

        expected_keys = ["done", "action", "action_value"]
        if from_pixels:
            expected_keys += ["pixels", "pixels_orig"]
        else:
            expected_keys += ["observation_orig", "observation_vector"]

        if not distributional:
            expected_keys += ["chosen_action_value"]
        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [("from_pixels=True", "catframes=4"), tuple()])
@pytest.mark.parametrize("gsde", [tuple(), ("gSDE=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_ddpg_maker(device, from_pixels, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    device = torch.device("cpu")
    flags = list(from_pixels + gsde)

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DDPGModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker()
        actor, value = make_ddpg_actor(proof_environment, device=device, cfg=cfg)
        td = proof_environment.reset().to(device)
        with set_exploration_mode(exploration):
            if UNSQUEEZE_SINGLETON and not td.ndimension():
                # Linear and conv used to break for non-batched data
                actor(td.unsqueeze(0))
            else:
                actor(td)
        expected_keys = ["done", "action", "param"]
        if from_pixels:
            expected_keys += ["pixels", "hidden", "pixels_orig"]
        else:
            expected_keys += ["observation_vector", "observation_orig"]

        if cfg.gSDE:
            expected_keys += ["scale", "loc", "_eps_gSDE"]

        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if cfg.gSDE:
            tsf_loc = actor.module[-1].module.transform(td.get("loc"))
            if exploration == "random":
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td.get("action"), tsf_loc)

        if UNSQUEEZE_SINGLETON and not td.ndimension():
            # Linear and conv used to break for non-batched data
            value(td.unsqueeze(0))
        else:
            value(td)
        expected_keys += ["state_action_value"]
        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        proof_environment.close()
        del proof_environment


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [tuple(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize("gsde", [tuple(), ("gSDE=True",)])
@pytest.mark.parametrize("shared_mapping", [tuple(), ("shared_mapping=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_ppo_maker(device, from_pixels, shared_mapping, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + shared_mapping + gsde)
    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            PPOModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)
        # if gsde and from_pixels:
        #     pytest.skip("gsde and from_pixels are incompatible")

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker()

        if cfg.from_pixels and not cfg.shared_mapping:
            with pytest.raises(
                RuntimeError,
                match="PPO learnt from pixels require the shared_mapping to be set to True",
            ):
                actor_value = make_ppo_model(
                    proof_environment,
                    device=device,
                    cfg=cfg,
                )
            return

        actor_value = make_ppo_model(
            proof_environment,
            device=device,
            cfg=cfg,
        )
        actor = actor_value.get_policy_operator()
        expected_keys = [
            "done",
            "pixels" if len(from_pixels) else "observation_vector",
            "pixels_orig" if len(from_pixels) else "observation_orig",
            "action",
            "sample_log_prob",
            "loc",
            "scale",
        ]
        if shared_mapping:
            expected_keys += ["hidden"]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        td = proof_environment.reset().to(device)
        td_clone = td.clone()
        with set_exploration_mode(exploration):
            if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
                # Linear and conv used to break for non-batched data
                actor(td_clone.unsqueeze(0))
            else:
                actor(td_clone)

        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if cfg.gSDE:
            if cfg.shared_mapping:
                tsf_loc = actor[-1].module[-1].module.transform(td_clone.get("loc"))
            else:
                tsf_loc = actor.module[-1].module.transform(td_clone.get("loc"))

            if exploration == "random":
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td_clone.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td_clone.get("action"), tsf_loc)

        value = actor_value.get_value_operator()
        expected_keys = [
            "done",
            "pixels" if len(from_pixels) else "observation_vector",
            "pixels_orig" if len(from_pixels) else "observation_orig",
            "state_value",
        ]
        if shared_mapping:
            expected_keys += ["hidden"]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        td_clone = td.clone()
        if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
            # Linear and conv used to break for non-batched data
            value(td_clone.unsqueeze(0))
        else:
            value(td_clone)
        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()
        del proof_environment


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [tuple(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize("gsde", [tuple(), ("gSDE=True",)])
@pytest.mark.parametrize("shared_mapping", [tuple(), ("shared_mapping=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_a2c_maker(device, from_pixels, shared_mapping, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + shared_mapping + gsde)
    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            A2CLossConfig,
            A2CModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)

    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)
        # if gsde and from_pixels:
        #     pytest.skip("gsde and from_pixels are incompatible")

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker()

        if cfg.from_pixels and not cfg.shared_mapping:
            with pytest.raises(
                RuntimeError,
                match="A2C learnt from pixels require the shared_mapping to be set to True",
            ):
                actor_value = make_a2c_model(
                    proof_environment,
                    device=device,
                    cfg=cfg,
                )
            return

        actor_value = make_a2c_model(
            proof_environment,
            device=device,
            cfg=cfg,
        )
        actor = actor_value.get_policy_operator()
        expected_keys = [
            "done",
            "pixels" if len(from_pixels) else "observation_vector",
            "pixels_orig" if len(from_pixels) else "observation_orig",
            "action",
            "sample_log_prob",
            "loc",
            "scale",
        ]
        if shared_mapping:
            expected_keys += ["hidden"]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        td = proof_environment.reset().to(device)
        td_clone = td.clone()
        with set_exploration_mode(exploration):
            if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
                # Linear and conv used to break for non-batched data
                actor(td_clone.unsqueeze(0))
            else:
                actor(td_clone)

        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if cfg.gSDE:
            if cfg.shared_mapping:
                tsf_loc = actor[-1].module[-1].module.transform(td_clone.get("loc"))
            else:
                tsf_loc = actor.module[-1].module.transform(td_clone.get("loc"))

            if exploration == "random":
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td_clone.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td_clone.get("action"), tsf_loc)

        value = actor_value.get_value_operator()
        expected_keys = [
            "done",
            "pixels" if len(from_pixels) else "observation_vector",
            "pixels_orig" if len(from_pixels) else "observation_orig",
            "state_value",
        ]
        if shared_mapping:
            expected_keys += ["hidden"]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        td_clone = td.clone()
        if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
            # Linear and conv used to break for non-batched data
            value(td_clone.unsqueeze(0))
        else:
            value(td_clone)
        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()
        del proof_environment

        with open_dict(cfg):
            cfg.advantage_in_loss = True
            loss_fn = make_a2c_loss(actor_value, cfg)
            cfg.advantage_in_loss = False
            loss_fn = make_a2c_loss(actor_value, cfg)


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("gsde", [tuple(), ("gSDE=True",)])
@pytest.mark.parametrize("from_pixels", [tuple()])
@pytest.mark.parametrize("tanh_loc", [tuple(), ("tanh_loc=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_sac_make(device, gsde, tanh_loc, from_pixels, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(gsde + tanh_loc + from_pixels)
    if gsde and from_pixels:
        pytest.skip("gsde and from_pixels are incompatible")

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            SACModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        if from_pixels:
            cfg.catframes = 4

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker()

        model = make_sac_model(
            proof_environment,
            device=device,
            cfg=cfg,
        )

        actor, qvalue, value = model
        td = proof_environment.reset().to(device)
        td_clone = td.clone()
        with set_exploration_mode(exploration):
            if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
                # Linear and conv used to break for non-batched data
                actor(td_clone.unsqueeze(0))
            else:
                actor(td_clone)

        expected_keys = [
            "done",
            "pixels" if len(from_pixels) else "observation_vector",
            "pixels_orig" if len(from_pixels) else "observation_orig",
            "action",
            "loc",
            "scale",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        if cfg.gSDE:
            tsf_loc = actor.module[-1].module.transform(td_clone.get("loc"))
            if exploration == "random":
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td_clone.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td_clone.get("action"), tsf_loc)

        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if UNSQUEEZE_SINGLETON and not td_clone.ndimension():
            # Linear and conv used to break for non-batched data
            qvalue(td_clone.unsqueeze(0))
        else:
            qvalue(td_clone)

        expected_keys = [
            "done",
            "observation_vector",
            "observation_orig",
            "action",
            "state_action_value",
            "loc",
            "scale",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        try:
            _assert_keys_match(td_clone, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if UNSQUEEZE_SINGLETON and not td.ndimension():
            # Linear and conv used to break for non-batched data
            value(td.unsqueeze(0))
        else:
            value(td)
        expected_keys = [
            "done",
            "observation_vector",
            "observation_orig",
            "state_value",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]

        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()
        del proof_environment


@pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [tuple(), ("from_pixels=True", "catframes=4")])
@pytest.mark.parametrize("gsde", [tuple(), ("gSDE=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_redq_make(device, from_pixels, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + gsde)
    if gsde and from_pixels:
        pytest.skip("gsde and from_pixels are incompatible")

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            REDQModelConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)

        env_maker = (
            ContinuousActionConvMockEnvNumpy
            if from_pixels
            else ContinuousActionVecMockEnv
        )
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker()

        model = make_redq_model(
            proof_environment,
            device=device,
            cfg=cfg,
        )
        actor, qvalue = model
        td = proof_environment.reset().to(device)
        with set_exploration_mode(exploration):
            actor(td)
        expected_keys = [
            "done",
            "action",
            "sample_log_prob",
            "loc",
            "scale",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]
        if from_pixels:
            expected_keys += ["hidden", "pixels", "pixels_orig"]
        else:
            expected_keys += ["observation_vector", "observation_orig"]

        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise

        if cfg.gSDE:
            tsf_loc = actor.module[-1].module.transform(td.get("loc"))
            if exploration == "random":
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(td.get("action"), tsf_loc)
            else:
                torch.testing.assert_close(td.get("action"), tsf_loc)

        qvalue(td)
        expected_keys = [
            "done",
            "action",
            "sample_log_prob",
            "state_action_value",
            "loc",
            "scale",
        ]
        if len(gsde):
            expected_keys += ["_eps_gSDE"]
        if from_pixels:
            expected_keys += ["hidden", "pixels", "pixels_orig"]
        else:
            expected_keys += ["observation_vector", "observation_orig"]
        try:
            _assert_keys_match(td, expected_keys)
        except AssertionError:
            proof_environment.close()
            raise
        proof_environment.close()
        del proof_environment


@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11.0"),
    reason="""Dreamer works with batches of null to 2 dimensions. Torch < 1.11
requires one-dimensional batches (for RNN and Conv nets for instance). If you'd like
to see torch < 1.11 supported for dreamer, please submit an issue.""",
)
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("tanh_loc", [tuple(), ("tanh_loc=True",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_dreamer_make(device, tanh_loc, exploration, dreamer_constructor_fixture):

    transformed_env_constructor = dreamer_constructor_fixture
    flags = ["from_pixels=True", "catframes=1"]

    config_fields = [
        (config_field.name, config_field.type, config_field)
        for config_cls in (
            EnvConfig,
            DreamerConfig,
        )
        for config_field in dataclasses.fields(config_cls)
    ]

    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=flags)
        env_maker = ContinuousActionConvMockEnvNumpy
        env_maker = transformed_env_constructor(
            cfg, use_env_creator=False, custom_env_maker=env_maker
        )
        proof_environment = env_maker().to(device)
        model = make_dreamer(
            proof_environment=proof_environment,
            device=device,
            cfg=cfg,
        )
        world_model, model_based_env, actor_model, value_model, policy = model
        out = world_model(proof_environment.rollout(3))
        expected_keys = {
            "action",
            "belief",
            "done",
            ("next", "belief"),
            ("next", "encoded_latents"),
            ("next", "pixels"),
            ("next", "pixels_orig"),
            ("next", "posterior_mean"),
            ("next", "posterior_std"),
            ("next", "prior_mean"),
            ("next", "prior_std"),
            ("next", "state"),
            "pixels",
            "pixels_orig",
            "reward",
            "state",
            ("next", "reco_pixels"),
            "next",
        }
        assert set(out.keys(True)) == expected_keys

        simulated_data = model_based_env.rollout(3)
        expected_keys = {
            "action",
            "belief",
            "done",
            ("next", "belief"),
            ("next", "state"),
            ("next", "pixels"),
            ("next", "pixels_orig"),
            "pixels_orig",
            "pixels",
            "reward",
            "state",
            "next",
        }
        assert expected_keys == set(simulated_data.keys(True))

        simulated_action = actor_model(model_based_env.reset())
        real_action = actor_model(proof_environment.reset())
        simulated_policy_action = policy(model_based_env.reset())
        real_policy_action = policy(proof_environment.reset())
        assert "action" in simulated_action.keys()
        assert "action" in real_action.keys()
        assert "action" in simulated_policy_action.keys()
        assert "action" in real_policy_action.keys()

        value_td = value_model(proof_environment.reset())
        assert "state_value" in value_td.keys()


@pytest.mark.parametrize("initial_seed", range(5))
def test_seed_generator(initial_seed):
    num_seeds = 100

    # Check unique seed generation
    if initial_seed == 0:
        with pytest.raises(ValueError):
            generate_seeds(initial_seed - 1, num_seeds)
        return
    else:
        seeds0 = generate_seeds(initial_seed - 1, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert len(seeds1) == num_seeds
    assert len(seeds1) == len(set(seeds1))
    assert len(set(seeds0).intersection(set(seeds1))) == 0

    # Check deterministic seed generation
    seeds0 = generate_seeds(initial_seed, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert seeds0 == seeds1


def test_timeit():
    n1 = 500
    w1 = 1e-4
    n2 = 200
    w2 = 1e-4
    w3 = 1e-4
    # warmup
    for _ in range(10):
        sleep(w1)
    for _ in range(n1):
        with timeit("event1"):
            sleep(w1)
        sleep(w3)
    for _ in range(n2):
        with timeit("event2"):
            sleep(w2)
    val1 = timeit._REG["event1"]
    val2 = timeit._REG["event2"]
    assert abs(val1[0] - w1) < 1e-2
    assert abs(val1[1] - n1 * w1) < 1
    assert val1[2] == n1
    assert abs(val2[0] - w2) < 1e-2
    assert abs(val2[1] - n2 * w2) < 1
    assert val2[2] == n2


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
