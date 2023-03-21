from copy import deepcopy

import torch.nn
import torch.optim
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
    CatFrames,
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    Resize,
    RewardScaling,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import (
    AdditiveGaussianWrapper,
    DdpgCnnActor,
    DdpgCnnQNet,
    DdpgMlpActor,
    DdpgMlpQNet,
    NoisyLinear,
    OrnsteinUhlenbeckProcessWrapper,
    ProbabilisticActor,
    TanhDelta,
    ValueOperator,
)
from torchrl.objectives import DDPGLoss, SoftUpdate
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers import Recorder
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
    if from_pixels is None:
        from_pixels = env_cfg.from_pixels

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
    from_pixels = env_cfg.from_pixels
    if from_pixels:
        return make_transformed_env_pixels(base_env, env_cfg)
    else:
        return make_transformed_env_states(base_env, env_cfg)


def make_transformed_env_pixels(base_env, env_cfg):
    if not isinstance(env_cfg.reward_scaling, float):
        env_cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(env_cfg.env_name, 5.0)

    env_library = LIBS[env_cfg.env_library]
    env = TransformedEnv(base_env)

    reward_scaling = env_cfg.reward_scaling

    env.append_transform(RewardScaling(0.0, reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []

    #
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))

    obs_norm = ObservationNorm(in_keys=["pixels"])
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


def make_transformed_env_states(base_env, env_cfg):
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


def get_stats(env_cfg):
    from_pixels = env_cfg.from_pixels
    env = make_transformed_env(make_base_env(env_cfg), env_cfg)
    init_stats(env, env_cfg.n_samples_stats, from_pixels)
    return env.state_dict()


def init_stats(env, n_samples_stats, from_pixels):
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            if from_pixels:
                t.init_stats(
                    n_samples_stats,
                    cat_dim=-3,
                    reduce_dim=(-1, -2, -3),
                    keep_dims=(-1, -2, -3),
                )
            else:
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
        devices=collector_cfg.collector_devices,
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


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_ddpg_model(cfg):

    env_cfg = cfg.env
    model_cfg = cfg.model
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)
    # we must initialize the observation norm transform
    init_stats(proof_environment, n_samples_stats=3, from_pixels=env_cfg.from_pixels)

    env_specs = proof_environment.specs
    from_pixels = env_cfg.from_pixels

    if not from_pixels:
        actor_net, q_net = make_ddpg_modules_state(model_cfg, proof_environment)
        in_keys = ["observation_vector"]
        out_keys = ["param"]
    else:
        actor_net, q_net = make_ddpg_modules_pixels(model_cfg, proof_environment)
        in_keys = ["pixels"]
        out_keys = ["param", "hidden"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

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

    if not from_pixels:
        in_keys = ["observation_vector", "action"]
    else:
        in_keys = ["pixels", "action"]

    out_keys = ["state_action_value"]
    value = ValueOperator(
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


def make_ddpg_modules_state(model_cfg, proof_environment):

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
    actor_net = DdpgMlpActor(**actor_net_default_kwargs)

    # Value model: DdpgMlpQNet is a specialized class that reads the state and
    # the action and outputs a value from it. It has two sub-components that
    # we parameterize with `mlp_net_kwargs_net1` and `mlp_net_kwargs_net2`.
    value_net_default_kwargs1 = {
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
    q_net = DdpgMlpQNet(
        mlp_net_kwargs_net1=value_net_default_kwargs1,
        mlp_net_kwargs_net2=value_net_default_kwargs2,
    )
    return actor_net, q_net


def make_ddpg_modules_pixels(model_cfg, proof_environment):
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
        "conv_net_kwargs": {"activation_class": ACTIVATIONS[model_cfg.activation]},
    }
    actor_net = DdpgCnnActor(**actor_net_default_kwargs)

    value_net_default_kwargs = {
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[model_cfg.activation],
        }
    }
    q_net = DdpgCnnQNet(**value_net_default_kwargs)

    return actor_net, q_net


def make_policy(model_cfg, actor):
    if model_cfg.ou_exploration:
        return OrnsteinUhlenbeckProcessWrapper(actor)
    else:
        return AdditiveGaussianWrapper(actor)


# ====================================================================
# DDPG Loss
# ---------


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


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(logger_cfg):
    exp_name = generate_exp_name("DDPG", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(
        logger_cfg.backend, logger_name="ddpg", experiment_name=exp_name
    )
    return logger


def make_recorder(cfg, logger, policy) -> Recorder:
    env_cfg = deepcopy(cfg.env)
    env = make_transformed_env(make_base_env(env_cfg, from_pixels=True), env_cfg)
    if cfg.recorder.video:
        env.insert_transform(
            0, VideoRecorder(logger=logger, tag=cfg.logger.exp_name, in_keys=["pixels"])
        )
    return Recorder(
        record_interval=1,
        record_frames=cfg.recorder.frames,
        frame_skip=env_cfg.frame_skip,
        policy_exploration=policy,
        recorder=env,
        exploration_mode="mean",
    )
