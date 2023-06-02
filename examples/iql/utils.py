import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    RenameTransform,
    RewardScaling,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import IQLLoss, SoftUpdate

from torchrl.record.loggers import generate_exp_name, get_logger

from torchrl.trainers.helpers.envs import LIBS
from torchrl.trainers.helpers.models import ACTIVATIONS


DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}

# ====================================================================
# Environment utils
# -----------------


def make_base_env(env_cfg, from_pixels=None):
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
    return make_transformed_env_states(base_env, env_cfg)


def make_transformed_env_states(base_env, env_cfg):
    if not isinstance(env_cfg.reward_scaling, float):
        env_cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(env_cfg.env_name, 1.0)

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


def make_test_env(env_cfg, state_dict):
    env_cfg.num_envs = 1
    env = make_parallel_env(env_cfg, state_dict=state_dict)
    return env


def get_stats(env_cfg):
    env = make_transformed_env(make_base_env(env_cfg), env_cfg)
    init_stats(env, env_cfg.n_samples_stats)
    return env.state_dict()


def init_stats(env, n_samples_stats):
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(n_samples_stats)


# ====================================================================
# Collector and replay buffer
# ---------------------------


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
        device=collector_cfg.collector_devices,
        init_random_frames=collector_cfg.init_random_frames,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
    )
    return collector


def make_replay_buffer(rb_cfg):
    if rb_cfg.prb:
        sampler = PrioritizedSampler(max_capacity=rb_cfg.capacity, alpha=0.7, beta=0.5)
    else:
        sampler = RandomSampler()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(rb_cfg.capacity), sampler=sampler
    )


def make_offline_replay_buffer(rb_cfg, state_dict):
    data = D4RLExperienceReplay(
        rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
    )
    data.append_transform(
        RewardScaling(
            loc=state_dict["transforms.0.loc"],
            scale=state_dict["transforms.0.scale"],
            standard_normal=state_dict["transforms.0.standard_normal"],
        )
    )
    data.append_transform(
        RenameTransform(
            ["observation", ("next", "observation")],
            ["observation_vector", ("next", "observation_vector")],
        )
    )
    data.append_transform(
        ObservationNorm(
            in_keys=["observation_vector"],
            loc=state_dict["transforms.2.loc"],
            scale=state_dict["transforms.2.scale"],
            standard_normal=state_dict["transforms.2.standard_normal"],
        )
    )
    data.append_transform(
        DoubleToFloat(
            in_keys=["observation_vector", ("next", "observation_vector")],
            in_keys_inv=[],
        )
    )

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_iql_model(cfg):
    env_cfg = cfg.env
    model_cfg = cfg.model
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)
    # we must initialize the observation norm transform
    init_stats(proof_environment, n_samples_stats=3)

    action_spec = proof_environment.action_spec

    actor_net, q_net, value_net = make_iql_modules_state(model_cfg, proof_environment)
    in_keys = ["observation_vector"]
    out_keys = ["loc", "scale"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

    # We use a ProbabilisticActor to make sure that we map the
    # network output to the right space using a TanhDelta
    # distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        spec=action_spec,
        # safe=False,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
            "tanh_loc": False,
        },
    )

    in_keys = ["observation_vector", "action"]

    out_keys = ["state_action_value"]
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=q_net,
    )
    in_keys = ["observation_vector"]
    out_keys = ["state_value"]
    value_net = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=value_net,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(max_steps=1000)
        print(td)
        actor(td)
        qvalue(td)
        value_net(td)

    return actor, qvalue, value_net


def make_iql_modules_state(model_cfg, proof_environment):
    action_spec = proof_environment.action_spec

    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    actor_net = MLP(**actor_net_kwargs)
    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{model_cfg.default_policy_scale}",
        scale_lb=model_cfg.scale_lb,
    )
    actor_net = torch.nn.Sequential(actor_net, actor_extractor)

    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }

    q_net = MLP(**qvalue_net_kwargs)

    # Define Value Network
    value_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    value_net = MLP(**value_net_kwargs)

    return actor_net, q_net, value_net


# ====================================================================
# IQL Loss
# ---------


def make_loss(loss_cfg, actor_network, qvalue_network, value_network):
    loss = IQLLoss(
        actor_network,
        qvalue_network,
        value_network=value_network,
        gamma=loss_cfg.gamma,
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        expectile=loss_cfg.expectile,
    )
    target_net_updater = SoftUpdate(loss, tau=loss_cfg.tau)
    target_net_updater.init_()
    return loss, target_net_updater


def make_iql_optimizer(optim_cfg, actor_network, qvalue_network, value_network):
    optim = torch.optim.Adam(
        list(actor_network.parameters())
        + list(qvalue_network.parameters())
        + list(value_network.parameters()),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optim


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(logger_cfg):
    exp_name = generate_exp_name("IQL", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(logger_cfg.backend, logger_name="iql", experiment_name=exp_name)
    return logger
