# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from sys import platform

import pytest
import torch.cuda
import torchrl.envs.transforms as T
import torchrl.objectives
from mocking_classes import ContinuousActionVecMockEnv
from torch import nn
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.dm_control import _has_dmc
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.modules import (
    TensorDictModule,
    DuelingMlpDQNet,
    QValueActor,
    ProbabilisticActor,
    MLP,
    NormalParamWrapper,
    ValueOperator,
    DdpgMlpActor,
    DdpgMlpQNet,
)
from torchrl.modules.models.models import _LAYER_CLASS_DICT
from torchrl.modules.tensordict_module.exploration import (
    EGreedyWrapper,
    AdditiveGaussianWrapper,
    OrnsteinUhlenbeckProcessWrapper,
)
from torchrl.trainers.loggers.csv import CSVLogger
from torchrl.trainers.loggers.mlflow import MLFlowLogger, _has_mlflow
from torchrl.trainers.loggers.tensorboard import TensorboardLogger, _has_tb
from torchrl.trainers.loggers.wandb import WandbLogger, _has_wandb

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False

if _has_mlflow:
    import mlflow

IS_OSX = platform == "darwin"


def make_env():
    def fun():
        return ContinuousActionVecMockEnv()

    return fun


@pytest.fixture(scope="module", autouse=True)
def init_hydra():
    GlobalHydra.instance().clear()
    hydra.initialize("../examples/configs/")
    yield
    GlobalHydra.instance().clear()


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestConfigs:
    @pytest.mark.parametrize(
        "file,num_workers",
        [
            ("async_sync", 2),
            ("sync_single", 0),
            ("sync_sync", 2),
        ],
    )
    def test_collector_configs(self, file, num_workers):
        create_env = make_env()
        policy = TensorDictModule(
            nn.Linear(7, 7), in_keys=["observation"], out_keys=["action"]
        )

        cfg = hydra.compose(
            "config", overrides=[f"collector={file}", f"num_workers={num_workers}"]
        )

        if cfg.num_workers == 0:
            create_env_fn = create_env
        else:
            create_env_fn = [
                create_env,
            ] * cfg.num_workers
        collector_partial = instantiate(cfg.collector)
        collector = collector_partial(policy=policy, create_env_fn=create_env_fn)
        for data in collector:
            assert data.numel() == 200
            break
        collector.shutdown()

    @pytest.mark.skipif(not _has_gym, reason="No gym found")
    @pytest.mark.skipif(not _has_dmc, reason="No DMC found")
    @pytest.mark.parametrize(
        "file,from_pixels",
        [
            ("cartpole", True),
            ("cartpole", False),
            ("halfcheetah", True),
            ("halfcheetah", False),
            ("cheetah", True),
            # ("cheetah",False), # processes fail -- to be investigated
        ],
    )
    def test_env_configs(self, file, from_pixels):
        if from_pixels and torch.cuda.device_count() == 0:
            return pytest.skip("not testing pixel rendering without gpu")

        cfg = hydra.compose(
            "config", overrides=[f"env={file}", f"++env.env.from_pixels={from_pixels}"]
        )

        env = instantiate(cfg.env)()

        tensordict = env.rollout(3)
        if from_pixels:
            assert "next_pixels" in tensordict.keys()
            assert tensordict["next_pixels"].shape[-1] == 3
        env.rollout(3)
        env.close()
        del env

    @pytest.mark.skipif(not _has_gym, reason="No gym found")
    @pytest.mark.skipif(not _has_dmc, reason="No DMC found")
    @pytest.mark.parametrize(
        "col_env_config",
        ["parallel_batch", "parallel", "single"],
    )
    def test_collection_env_configs(self, col_env_config):
        cfg = hydra.compose(
            "config",
            overrides=[
                f"collection_env={col_env_config}",
            ],
        )

        # env = instantiate(cfg).collection_env
        env = instantiate(cfg.collection_env)
        print(env)
        del env

    @pytest.mark.skipif(not _has_gym, reason="No gym found")
    @pytest.mark.skipif(not _has_dmc, reason="No gym found")
    @pytest.mark.parametrize(
        "env_file,transform_file",
        [
            ["cartpole", "pixels"],
            ["halfcheetah", "pixels"],
            # ["cheetah", "pixels"],
            ["cartpole", "state"],
            ["halfcheetah", "state"],
            ["cheetah", "state"],
        ],
    )
    def test_transforms_configs(self, env_file, transform_file):
        if transform_file == "state":
            from_pixels = False
        else:
            if torch.cuda.device_count() == 0:
                return pytest.skip("not testing pixel rendering without gpu")
            from_pixels = True
        cfg = hydra.compose(
            "config",
            overrides=[
                f"env={env_file}",
                f"++env.env.from_pixels={from_pixels}",
                f"transforms={transform_file}",
            ],
        )

        env = instantiate(cfg.env)()
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)
        env.rollout(3)
        env.close()
        del env

    @pytest.mark.parametrize(
        "file",
        [
            "circular",
            "prioritized",
        ],
    )
    @pytest.mark.parametrize(
        "size",
        [
            "10",
            None,
        ],
    )
    def test_replaybuffer(self, file, size):
        args = [f"replay_buffer={file}"]
        if size is not None:
            args += [f"replay_buffer.size={size}"]
        cfg = hydra.compose("config", overrides=args)
        replay_buffer = instantiate(cfg.replay_buffer)
        assert replay_buffer._capacity == replay_buffer._storage.max_size


def make_actor_dqn(net_partial, actor_partial, env, out_features=None):
    if out_features is not None:
        out_features = [out_features] + list(env.action_spec.shape)
    else:
        out_features = list(env.action_spec.shape)
    network = net_partial.network(out_features=out_features)
    actor = actor_partial.actor(
        module=network, in_keys=net_partial.in_keys, spec=env.action_spec
    )
    return actor


def make_model_ppo(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(
        out_features=out_features, spec=env.action_spec
    )
    actor_critic = net_partial.actor_critic(policy_operator=policy_operator)
    return actor_critic


def make_model_sac(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(
        out_features=out_features, spec=env.action_spec
    )

    qvalue_operator = net_partial.qvalue_network

    value_operator = net_partial.value_network
    return policy_operator, qvalue_operator, value_operator


def make_model_ddpg(net_partial, env):
    out_features = env.action_spec.shape[-1]

    # build the module
    policy_operator = net_partial.policy_network(
        out_features=out_features, spec=env.action_spec
    )

    qvalue = net_partial.value_operator
    return policy_operator, qvalue


def make_model_redq(net_partial, model_params, env):
    out_features = env.action_spec.shape[-1] * model_params.out_features_multiplier

    # build the module
    policy_operator = net_partial.policy_network(
        out_features=out_features, spec=env.action_spec
    )

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
        env = instantiate(cfg.env)()
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
        env = instantiate(cfg.env)()
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
        env = instantiate(cfg.env)()
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
        env = instantiate(cfg.env)()
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
        env = instantiate(cfg.env)()
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


def make_halfcheetah_env_with_state_transforms():
    transforms = T.Compose(
        T.ObservationNorm(0, 1, ["next_observation"]),
        T.RewardScaling(0, 1, ["reward"]),
        T.CatTensors(),
        T.DoubleToFloat(["next_observation_vector"]),
    )
    env = TransformedEnv(GymEnv("HalfCheetah-v4"), transforms)
    return env


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestExplorationConfigs:
    _WRAPPER_MAP = {
        "e_greedy": EGreedyWrapper,
        "additive_gaussian": AdditiveGaussianWrapper,
        "ou_process": OrnsteinUhlenbeckProcessWrapper,
    }

    @pytest.mark.parametrize("network", ["linear", "noisy_linear"])
    @pytest.mark.parametrize(
        "wrapper", [None, "e_greedy", "additive_gaussian", "ou_process"]
    )
    def test_exploration(self, network, wrapper):
        additional_config = [
            "env=halfcheetah",
            "transforms=state",
            "model=sac/discrete",
            "network=sac/independent_state",
        ]
        exploration_config = []
        if wrapper is not None:
            exploration_config += [
                f"exploration={wrapper}",
                f"exploration.network={network}",
            ]
        else:
            exploration_config += [f"exploration={network}"]

        cfg = hydra.compose("config", overrides=exploration_config + additional_config)
        env = make_halfcheetah_env_with_state_transforms()
        model_params = instantiate(cfg.model)
        net_partial = instantiate(cfg.network)
        actor, qvalue, value = make_model_sac(net_partial, model_params, env)
        actor(env.reset())
        assert actor.module.module.layer_class is _LAYER_CLASS_DICT[network]

        if cfg.exploration.exploration_wrapper is not None:
            actor_explore = instantiate(cfg.exploration.exploration_wrapper)(actor)
            assert type(actor_explore) is self._WRAPPER_MAP[wrapper]
        else:
            actor_explore = actor

        rollout = env.rollout(3)
        assert all(key in rollout.keys() for key in actor_explore.in_keys), (
            actor_explore.in_keys,
            rollout.keys(),
        )
        tensordict = env.rollout(3, actor_explore)
        assert env.action_spec.is_in(tensordict["action"]), env.action_spec
        qvalue(tensordict)
        value(tensordict)


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestLossConfigs:
    def _make_actor_dqn(self, env):
        network = DuelingMlpDQNet(
            out_features=env.action_spec.shape[-1],
            mlp_kwargs_output={"num_cells": 10, "layer_class": "linear"},
        )
        actor = QValueActor(
            module=network, in_keys=["observation_vector"], spec=env.action_spec
        )
        return actor

    def _make_model_ppo(self, env):
        td_module = TensorDictModule(
            module=NormalParamWrapper(
                MLP(num_cells=[10], out_features=2 * env.action_spec.shape[-1])
            ),
            in_keys=["observation_vector"],
            out_keys=["loc", "scale"],
        )
        policy_operator = ProbabilisticActor(
            module=td_module,
            dist_param_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class="TanhNormal",
        )
        value_operator = ValueOperator(
            MLP(num_cells=[10], out_features=1), in_keys=["observation_vector"]
        )
        return policy_operator, value_operator

    def _make_model_sac(self, env):
        td_module = TensorDictModule(
            module=NormalParamWrapper(
                MLP(num_cells=[10], out_features=2 * env.action_spec.shape[-1])
            ),
            in_keys=["observation_vector"],
            out_keys=["loc", "scale"],
        )
        policy_operator = ProbabilisticActor(
            module=td_module,
            dist_param_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class="TanhNormal",
        )
        qvalue_operator = ValueOperator(
            MLP(num_cells=[10], out_features=1),
            in_keys=["action", "observation_vector"],
        )
        value_operator = ValueOperator(
            MLP(num_cells=[10], out_features=1), in_keys=["observation_vector"]
        )
        return policy_operator, qvalue_operator, value_operator

    def _make_model_ddpg(self, env):
        td_module = TensorDictModule(
            module=DdpgMlpActor(
                mlp_net_kwargs={"num_cells": 10}, out_features=env.action_spec.shape[-1]
            ),
            in_keys=["observation_vector"],
            out_keys=["param"],
        )
        policy_operator = ProbabilisticActor(
            module=td_module,
            dist_param_keys=["param"],
            spec=env.action_spec,
            distribution_class="TanhDelta",
        )
        qvalue_operator = ValueOperator(
            DdpgMlpQNet(), in_keys=["observation_vector", "action"]
        )
        return policy_operator, qvalue_operator

    def _make_model_redq(self, env):
        td_module = TensorDictModule(
            module=NormalParamWrapper(
                MLP(num_cells=[10], out_features=2 * env.action_spec.shape[-1])
            ),
            in_keys=["observation_vector"],
            out_keys=["loc", "scale"],
        )
        policy_operator = ProbabilisticActor(
            module=td_module,
            dist_param_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class="TanhNormal",
        )
        qvalue_operator = ValueOperator(
            MLP(num_cells=[10], out_features=1),
            in_keys=["action", "observation_vector"],
        )
        return policy_operator, qvalue_operator

    def test_ddpg_loss(self):
        config = ["loss=ddpg_loss"]
        cfg = hydra.compose("config", overrides=config)
        env = make_halfcheetah_env_with_state_transforms()
        loss_partial = instantiate(cfg.loss)
        actor, qvalue = self._make_model_ddpg(env)
        qvalue(actor(env.reset()))
        loss = loss_partial(actor, qvalue)

        assert isinstance(loss, torchrl.objectives.DDPGLoss)
        for param in ["gamma", "loss_function", "delay_actor", "delay_value"]:
            assert cfg.loss[param] == getattr(loss, param)

    def test_dqn_loss(self):
        config = ["loss=dqn_loss"]
        cfg = hydra.compose("config", overrides=config)
        env = make_halfcheetah_env_with_state_transforms()
        loss_partial = instantiate(cfg.loss)
        actor = self._make_actor_dqn(env)
        actor(env.reset())
        loss = loss_partial(actor)

        assert isinstance(loss, torchrl.objectives.DQNLoss)
        for param in ["gamma", "loss_function", "priority_key", "delay_value"]:
            assert cfg.loss[param] == getattr(loss, param)

    def test_ppo_loss(self):
        config = ["loss=ppo_loss"]
        cfg = hydra.compose("config", overrides=config)
        env = make_halfcheetah_env_with_state_transforms()
        loss_partial = instantiate(cfg.loss)
        actor, critic = self._make_model_ppo(env)
        critic(actor(env.reset()))
        loss = loss_partial(actor, critic)

        assert isinstance(loss, torchrl.objectives.PPOLoss)
        for param in [
            "advantage_key",
            "advantage_diff_key",
            "samples_mc_entropy",
            "entropy_coef",
            "critic_coef",
            "gamma",
            "loss_critic_type",
        ]:
            assert cfg.loss[param] == getattr(loss, param)
        assert loss.entropy_bonus == (
            cfg.loss["entropy_bonus"] and cfg.loss["entropy_coef"]
        )
        # non-primitive loss.advantage_module is not tested

    def test_redq_loss(self):
        config = ["loss=redq_loss"]
        cfg = hydra.compose("config", overrides=config)
        env = make_halfcheetah_env_with_state_transforms()
        loss_partial = instantiate(cfg.loss)
        actor, qvalue = self._make_model_redq(env)
        qvalue(actor(env.reset()))
        loss = loss_partial(actor, qvalue)

        assert isinstance(loss, torchrl.objectives.REDQLoss)
        for param in [
            "num_qvalue_nets",
            "gamma",
            "priority_key",
            "loss_function",
            "alpha_init",
            "delay_qvalue",
            "gSDE",
        ]:
            assert cfg.loss[param] == getattr(loss, param)
        assert (
            max(1, min(cfg.loss["sub_sample_len"], cfg.loss["num_qvalue_nets"] - 1))
            == loss.sub_sample_len
        )
        assert math.log(cfg.loss["min_alpha"]) == loss.min_log_alpha
        assert math.log(cfg.loss["max_alpha"]) == loss.max_log_alpha
        assert math.log(cfg.loss["alpha_init"]) == loss.log_alpha
        if cfg.loss["target_entropy"] != "auto":
            assert cfg.loss["target_entropy"] == loss.target_entropy

    def test_sac_loss(self):
        config = ["loss=sac_loss"]
        cfg = hydra.compose("config", overrides=config)
        env = make_halfcheetah_env_with_state_transforms()
        loss_partial = instantiate(cfg.loss)
        actor, qvalue, value = self._make_model_sac(env)
        value(qvalue(actor(env.reset())))
        loss = loss_partial(actor, qvalue, value)

        assert isinstance(loss, torchrl.objectives.SACLoss)
        for param in [
            "num_qvalue_nets",
            "gamma",
            "priority_key",
            "loss_function",
            "alpha_init",
            "delay_actor",
            "delay_qvalue",
            "delay_value",
        ]:
            assert cfg.loss[param] == getattr(loss, param)
        assert math.log(cfg.loss["min_alpha"]) == loss.min_log_alpha
        assert math.log(cfg.loss["max_alpha"]) == loss.max_log_alpha
        assert math.log(cfg.loss["alpha_init"]) == loss.log_alpha
        if cfg.loss["target_entropy"] != "auto":
            assert cfg.loss["target_entropy"] == loss.target_entropy


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestLoggerConfigs:
    @pytest.fixture
    def mlflow_teardown(self):
        yield
        mlflow.end_run()

    def test_csv_logger(self, tmp_path):
        config = ["logger=csv", f"++logger.log_dir={tmp_path}"]
        cfg = hydra.compose("config", overrides=config)
        logger = instantiate(cfg.logger)
        assert isinstance(logger, CSVLogger)

    @pytest.mark.skipif(not _has_mlflow, reason="No mlflow found")
    def test_mlflow_logger(self, tmp_path, mlflow_teardown):
        config = ["logger=mlflow", f"++logger.tracking_uri={tmp_path}"]
        cfg = hydra.compose("config", overrides=config)
        logger = instantiate(cfg.logger)
        assert isinstance(logger, MLFlowLogger)

    @pytest.mark.skipif(not _has_tb, reason="No tensorboard found")
    def test_tensorboard_logger(self, tmp_path):
        config = ["logger=tensorboard", f"++logger.log_dir={tmp_path}"]
        cfg = hydra.compose("config", overrides=config)
        logger = instantiate(cfg.logger)
        assert isinstance(logger, TensorboardLogger)

    @pytest.mark.skipif(not _has_wandb, reason="No wandb found")
    def test_wandb_logger(self, tmp_path):
        # offline needs to be set for testing, otherwise login will be attempted
        config = [
            "logger=wandb",
            f"++logger.save_dir={tmp_path}",
            "++logger.offline=True",
        ]
        cfg = hydra.compose("config", overrides=config)
        logger = instantiate(cfg.logger)
        assert isinstance(logger, WandbLogger)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
