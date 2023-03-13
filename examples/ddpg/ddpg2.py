# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy

import hydra
import torch.cuda
import tqdm
from hydra.core.config_store import ConfigStore
from tensordict.nn import TensorDictModule
from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector
from torchrl.data import (
    CompositeSpec,
    LazyMemmapStorage,
    MultiStep,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import (
    AdditiveGaussianWrapper,
    DdpgMlpActor,
    DdpgMlpQNet,
    NoisyLinear,
    OrnsteinUhlenbeckProcessWrapper,
    ProbabilisticActor,
    SafeModule,
    TanhDelta,
    ValueOperator,
)
from torchrl.objectives import DDPGLoss, SoftUpdate
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger, WandbLogger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    EnvConfig,
    initialize_observation_norm_transforms,
    LIBS,
    parallel_env_constructor,
    retrieve_observation_norms_state_dict,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import LossConfig, make_ddpg_loss
from torchrl.trainers.helpers.models import (
    ACTIVATIONS,
    DDPGModelConfig,
    make_ddpg_actor,
)
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig


DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


def make_base_env(env_cfg, from_pixels=False):
    env_library = LIBS[env_cfg.env_library]
    env_name = env_cfg.env_name
    frame_skip = env_cfg.frame_skip

    env_kwargs = {
        "env_name": env_name,
        "frame_skip": frame_skip,
        "from_pixels": from_pixels,  # for rendering
        "pixels_only": False,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.env_task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    if env_cfg.noop > 1:
        env = TransformedEnv(env, NoopResetEnv(env_cfg.noop))
    return env


def make_transformed_env(base_env, env_cfg):
    if not isinstance(env_cfg.reward_scaling, float):
        env_cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(env_cfg.env_name, 5.0)

    env_library = LIBS[env_cfg.env_library]
    env = TransformedEnv(base_env)

    reward_scaling = env_cfg.reward_scaling

    env.append_transform(RewardScaling(0.0, reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []

    # we concatenate all the state vectors
    # even if there is a single tensor, it'll be renamed in "observation_vector"
    selected_keys = [
        key for key in env.observation_spec.keys(True, True) if key != "pixels"
    ]
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    obs_norm = ObservationNorm(in_keys=[out_key])
    env.append_transform(obs_norm)

    if env_library is DMControlEnv:
        double_to_float_list += [
            "reward",
        ]
        double_to_float_list += [
            "action",
        ]
        double_to_float_inv_list += ["action"]  # DMControl requires double-precision
        double_to_float_list += ["observation_vector"]
    else:
        double_to_float_list += ["observation_vector"]
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )
    return env


def make_parallel_env(env_cfg, state_dict):
    num_envs = env_cfg.num_envs
    env = make_transformed_env(
        ParallelEnv(num_envs, EnvCreator(lambda: make_base_env(env_cfg))), env_cfg
    )
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.load_state_dict(state_dict)
    return env


def make_collector(cfg, state_dict, policy):
    env_cfg = cfg.env
    loss_cfg = cfg.loss
    collector_cfg = cfg.collector
    if collector_cfg.async_collection:
        collector_class = MultiaSyncDataCollector
    else:
        collector_class = MultiSyncDataCollector
    if collector_cfg.multi_step:
        ms = MultiStep(gamma=loss_cfg.gamma, n_steps=collector_cfg.multi_step)
    else:
        ms = None
    collector = collector_class(
        [make_parallel_env(env_cfg, state_dict=state_dict)]
        * collector_cfg.num_collectors,
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        postproc=ms,
        devices=collector_cfg.collector_devices,
        init_random_frames=collector_cfg.init_random_frames,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
    )
    return collector


def make_logger(logger_cfg):
    if logger_cfg.logger_class == "wandb":
        logger = WandbLogger(logger_cfg.exp_name)
    else:
        raise NotImplementedError
    return logger


def make_recorder(cfg, logger):
    env_cfg = deepcopy(cfg.env)
    env = make_transformed_env(make_base_env(env_cfg, from_pixels=True), env_cfg)
    env.insert_transform(
        0, VideoRecorder(logger=logger, tag=cfg.logger.exp_name, in_keys=["pixels"])
    )


def make_replay_buffer(rb_cfg):
    if rb_cfg.prb:
        sampler = PrioritizedSampler(max_capacity=rb_cfg.capacity, alpha=0.7, beta=0.5)
    else:
        sampler = RandomSampler()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(rb_cfg.capacity), sampler=sampler
    )


def make_ddpg_model(cfg):

    env_cfg = cfg.env
    model_cfg = cfg.model
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)

    noisy = model_cfg.noisy

    linear_layer_class = torch.nn.Linear if not noisy else NoisyLinear

    env_specs = proof_environment.specs
    out_features = env_specs["input_spec"]["action"].shape[0]

    actor_net_default_kwargs = {
        "action_dim": out_features,
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[model_cfg.activation],
        },
    }
    in_keys = ["observation_vector"]
    actor_net = DdpgMlpActor(**actor_net_default_kwargs)
    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=["param"])

    # We use a ProbabilisticActor to make sure that we map the
    # network output to the right space using a TanhDelta
    # distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["param"],
        spec=CompositeSpec(action=env_specs["input_spec"]["action"]),
        safe=True,
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env_specs["input_spec"]["action"].space.minimum,
            "max": env_specs["input_spec"]["action"].space.maximum,
        },
    )

    # Value model: DdpgMlpQNet is a specialized class that reads the state and
    # the action and outputs a value from it. It has two sub-components that
    # we parameterize with `mlp_net_kwargs_net1` and `mlp_net_kwargs_net2`.
    state_class = ValueOperator
    value_net_default_kwargs1 = {
        "activation_class": ACTIVATIONS[model_cfg.activation],
        "layer_class": linear_layer_class,
        "activation_class": ACTIVATIONS[model_cfg.activation],
        "bias_last_layer": True,
    }
    value_net_default_kwargs2 = {
        "num_cells": [400, 300],
        "activation_class": ACTIVATIONS[model_cfg.activation],
        "bias_last_layer": True,
        "layer_class": linear_layer_class,
    }
    in_keys = ["observation_vector", "action"]
    out_keys = ["state_action_value"]
    q_net = DdpgMlpQNet(
        mlp_net_kwargs_net1=value_net_default_kwargs1,
        mlp_net_kwargs_net2=value_net_default_kwargs2,
    )
    value = state_class(
        in_keys=in_keys,
        out_keys=out_keys,
        module=q_net,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_mode("random"):
        for t in proof_environment.transform:
            if isinstance(t, ObservationNorm):
                t.init_stats(2)
        td = proof_environment.rollout(max_steps=1000)
        print(td)
        actor(td)
        value(td)

    return actor, value


def make_policy(model_cfg, actor):
    if model_cfg.ou_exploration:
        return OrnsteinUhlenbeckProcessWrapper(actor)
    else:
        return AdditiveGaussianWrapper(actor)


def get_stats(env_cfg):
    env = make_transformed_env(make_base_env(env_cfg), env_cfg)
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(env_cfg.n_samples_stats)
    return env.state_dict()


def make_loss(loss_cfg, actor_network, value_network):
    loss = DDPGLoss(
        actor_network,
        value_network,
        gamma=loss_cfg.gamma,
        loss_function=loss_cfg.loss_function,
    )
    target_net_updater = SoftUpdate(loss, 1 - loss_cfg.tau)
    target_net_updater.init_()
    return loss, target_net_updater


def make_optim(optim_cfg, actor_network, value_network):
    optim = torch.optim.Adam(
        list(actor_network.parameters()) + list(value_network.parameters()),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optim


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)
    model_device = cfg.optim.device

    exp_name = generate_exp_name("DDPG", cfg.logger.exp_name)

    state_dict = get_stats(cfg.env)
    logger = make_logger(cfg.logger)
    recorder = make_recorder(cfg, logger)
    replay_buffer = make_replay_buffer(cfg.replay_buffer)

    actor_network, value_network = make_ddpg_model(cfg)
    actor_network = actor_network.to(model_device)
    value_network = value_network.to(model_device)

    policy = make_policy(cfg.model, actor_network)
    collector = make_collector(cfg, state_dict=state_dict, policy=policy)
    loss, target_net_updater = make_loss(cfg.loss, actor_network, value_network)
    optim = make_optim(cfg.optim, actor_network, value_network)

    optim_steps_per_batch = cfg.optim.optim_steps_per_batch
    batch_size = cfg.optim.batch_size
    init_random_frames = cfg.collector.init_random_frames

    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_frames = 0
    for i, data in enumerate(collector):
        collected_frames += data.numel()
        pbar.update(data.numel())
        # extend replay buffer
        replay_buffer.extend(data.view(-1))
        if collected_frames >= init_random_frames:
            for j in range(optim_steps_per_batch):
                # sample
                sample = replay_buffer.sample(batch_size)
                # loss
                loss_vals = loss(sample)
                # backprop
                loss_val = sum(
                    val for key, val in loss_vals.items() if key.startswith("loss")
                )
                loss_val.backward()
                optim.step()
                optim.zero_grad()
                target_net_updater.step()
                pbar.set_description(f"loss: {loss_val.item(): 4.4f}")
            collector.update_policy_weights_()


if __name__ == "__main__":
    main()
