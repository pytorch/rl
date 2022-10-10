# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses

import pytest
import torch
from _utils_internal import get_available_devices, generate_seeds

try:
    from hydra import initialize, compose
    from hydra.core.config_store import ConfigStore

    _has_hydra = True
except ImportError:
    _has_hydra = False
from mocking_classes import (
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    DiscreteActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.utils import set_exploration_mode
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.helpers.envs import EnvConfig
from torchrl.trainers.helpers.models import (
    make_dqn_actor,
    make_ddpg_actor,
    make_ppo_model,
    make_sac_model,
    make_redq_model,
    DiscreteModelConfig,
    DDPGModelConfig,
    PPOModelConfig,
    SACModelConfig,
    REDQModelConfig,
)

## these tests aren't truly unitary but setting up a fake env for the
# purpose of building a model with args is a lot of unstable scaffoldings
# with unclear benefits


def _assert_keys_match(td, expeceted_keys):
    td_keys = list(td.keys())
    d = set(td_keys) - set(expeceted_keys)
    assert len(d) == 0, f"{d} is in tensordict but unexpected: {td.keys()}"
    d = set(expeceted_keys) - set(td_keys)
    assert len(d) == 0, f"{d} is expected but not in tensordict: {td.keys()}"
    assert len(td_keys) == len(expeceted_keys)


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.skipif(not _has_hydra, reason="No hydra library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("noisy", [tuple(), ("noisy=True",)])
@pytest.mark.parametrize("distributional", [tuple(), ("distributional=True",)])
@pytest.mark.parametrize("from_pixels", [tuple(), ("from_pixels=True", "catframes=4")])
def test_dqn_maker(device, noisy, distributional, from_pixels):
    flags = list(noisy + distributional + from_pixels) + ["env_name=CartPole-v1"]

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
        proof_environment = env_maker()

        actor = make_dqn_actor(proof_environment, cfg, device)
        td = proof_environment.reset().to(device)
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
