# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sys import platform

import pytest
import torch.cuda

try:
    import hydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False

from test.test_configs.test_configs_common import init_hydra  # noqa: F401

IS_OSX = platform == "darwin"


def make_actor_dqn(net_partial, actor_partial, env, out_features=None):
    if out_features is not None:
        out_features = [out_features] + list(env.action_spec.shape)
    else:
        out_features = list(env.action_spec.shape)
    network = net_partial.network(out_features=out_features)
    actor = actor_partial.actor(module=network, in_keys=net_partial.in_keys)
    return actor


def make_model_ppo(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(out_features=out_features)
    actor_critic = net_partial.actor_critic(policy_operator=policy_operator)
    return actor_critic


def make_model_sac(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(out_features=out_features)

    qvalue_operator = net_partial.qvalue_network

    value_operator = net_partial.value_network
    return policy_operator, qvalue_operator, value_operator


def make_model_ddpg(net_partial, env):
    out_features = env.action_spec.shape[-1]

    # build the module
    policy_operator = net_partial.policy_network(out_features=out_features)

    qvalue = net_partial.value_operator
    return policy_operator, qvalue


def make_model_redq(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(out_features=out_features)

    qvalue_operator = net_partial.qvalue_network

    return policy_operator, qvalue_operator


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestModelConfigs:
    @pytest.mark.parametrize("pixels", [True, False])
    @pytest.mark.parametrize("distributional", [True, False])
    def test_dqn(self, pixels, distributional):
        env_config = ["env=cartpole"]
        if pixels:
            net_conf = "network=dqn/pixels"
            env_config += ["transforms=pixels", "++env.env.from_pixels=True"]
        else:
            net_conf = "network=dqn/state"
            env_config += ["transforms=state"]
        if distributional:
            model_conf = "model=dqn/distributional"
        else:
            model_conf = "model=dqn/regular"

        cfg = hydra.compose("config", overrides=env_config + [net_conf] + [model_conf])
        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)

        actor_partial = instantiate(cfg.model)
        net_partial = instantiate(cfg.network)
        out_features = (
            cfg.model.out_features if hasattr(cfg.model, "out_features") else None
        )
        actor = make_actor_dqn(net_partial, actor_partial, env, out_features)
        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actor.in_keys), (
            actor.in_keys,
            rollout.keys(),
        )
        tensordict = actor(rollout)
        assert env.action_spec.is_in(tensordict["action"])

    @pytest.mark.parametrize("pixels", [True, False])
    @pytest.mark.parametrize("independent", [True, False])
    @pytest.mark.parametrize("continuous", [True, False])
    def test_ppo(self, pixels, independent, continuous):
        if IS_OSX and pixels and continuous:
            pytest.skip("rendering halfcheetah can throw gladLoadGL error on OSX")
        torch.manual_seed(0)
        env_config = []
        if independent:
            prefix = "independent"
        else:
            prefix = "shared"
        if pixels:
            suffix = "pixels"
            env_config += ["transforms=pixels", "++env.env.from_pixels=True"]
        else:
            suffix = "state"
            env_config += ["transforms=state"]
        net_conf = f"network=ppo/{prefix}_{suffix}"

        if continuous:
            env_config += ["env=halfcheetah"]
            model_conf = "model=ppo/continuous"
        else:
            env_config += ["env=cartpole"]
            model_conf = "model=ppo/discrete"

        cfg = hydra.compose("config", overrides=env_config + [net_conf] + [model_conf])
        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)

        model_params = instantiate(cfg.model)
        net_partial = instantiate(cfg.network)
        actorcritic = make_model_ppo(net_partial, model_params, env)
        actorcritic(env.reset())
        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actorcritic.in_keys), (
            actorcritic.in_keys,
            rollout.keys(),
        )
        tensordict = env.rollout(3, actorcritic)
        assert env.action_spec.is_in(tensordict["action"]), env.action_spec

    @pytest.mark.parametrize("pixels", [True, False])
    @pytest.mark.parametrize("independent", [True])
    @pytest.mark.parametrize("continuous", [True, False])
    def test_sac(self, pixels, independent, continuous):
        if IS_OSX and pixels and continuous:
            pytest.skip("rendering halfcheetah can throw gladLoadGL error on OSX")
        torch.manual_seed(0)
        env_config = []
        if independent:
            prefix = "independent"
        else:
            prefix = "shared"
        if pixels:
            suffix = "pixels"
            env_config += ["transforms=pixels", "++env.env.from_pixels=True"]
        else:
            suffix = "state"
            env_config += ["transforms=state"]
        net_conf = f"network=sac/{prefix}_{suffix}"

        if continuous:
            env_config += ["env=halfcheetah"]
            model_conf = "model=sac/continuous"
        else:
            env_config += ["env=cartpole"]
            model_conf = "model=sac/discrete"

        cfg = hydra.compose("config", overrides=env_config + [net_conf] + [model_conf])
        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)

        model_params = instantiate(cfg.model)
        net_partial = instantiate(cfg.network)
        actor, qvalue, value = make_model_sac(net_partial, model_params, env)
        actor(env.reset())
        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actor.in_keys), (
            actor.in_keys,
            rollout.keys(),
        )
        tensordict = env.rollout(3, actor)
        assert env.action_spec.is_in(tensordict["action"]), env.action_spec
        qvalue(tensordict)
        value(tensordict)

    @pytest.mark.parametrize("pixels", [True, False])
    def test_ddpg(
        self,
        pixels,
    ):
        if IS_OSX and pixels:
            pytest.skip("rendering halfcheetah can throw gladLoadGL error on OSX")
        torch.manual_seed(0)
        env_config = []
        if pixels:
            suffix = "pixels"
            env_config += ["transforms=pixels", "++env.env.from_pixels=True"]
        else:
            suffix = "state"
            env_config += ["transforms=state"]
        net_conf = f"network=ddpg/{suffix}"

        env_config += ["env=halfcheetah"]
        model_conf = "model=ddpg/basic"

        cfg = hydra.compose("config", overrides=env_config + [net_conf] + [model_conf])
        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)

        net_partial = instantiate(cfg.network)
        actor, qvalue = make_model_ddpg(net_partial, env)
        actor(env.reset())
        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actor.in_keys), (
            actor.in_keys,
            rollout.keys(),
        )
        tensordict = env.rollout(3, actor)
        assert env.action_spec.is_in(tensordict["action"]), env.action_spec
        qvalue(tensordict)

    @pytest.mark.parametrize("pixels", [True, False])
    @pytest.mark.parametrize("independent", [True])
    @pytest.mark.parametrize("continuous", [True, False])
    def test_redq(self, pixels, independent, continuous):
        if IS_OSX and pixels and continuous:
            pytest.skip("rendering halfcheetah can throw gladLoadGL error on OSX")
        torch.manual_seed(0)
        env_config = []
        if independent:
            prefix = "independent"
        else:
            prefix = "shared"
        if pixels:
            suffix = "pixels"
            env_config += ["transforms=pixels", "++env.env.from_pixels=True"]
        else:
            suffix = "state"
            env_config += ["transforms=state"]
        net_conf = f"network=redq/{prefix}_{suffix}"

        if continuous:
            env_config += ["env=halfcheetah"]
            model_conf = "model=redq/continuous"
        else:
            env_config += ["env=cartpole"]
            model_conf = "model=redq/discrete"

        cfg = hydra.compose("config", overrides=env_config + [net_conf] + [model_conf])
        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)

        model_params = instantiate(cfg.model)
        net_partial = instantiate(cfg.network)
        actor, qvalue = make_model_redq(net_partial, model_params, env)
        actor(env.reset())
        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actor.in_keys), (
            actor.in_keys,
            rollout.keys(),
        )
        tensordict = env.rollout(3, actor)
        assert env.action_spec.is_in(tensordict["action"]), env.action_spec
        qvalue(tensordict)
