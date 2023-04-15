from copy import deepcopy

import torch.nn
import torch.optim
import torch.distributions as dist
from tensordict.nn import TensorDictModule

from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    LazyMemmapStorage,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
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
    ConvNet,
    MLP,
    SafeModule,
    ProbabilisticActor,
    ValueOperator,
    ActorValueOperator,
    OneHotCategorical,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
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
    env_kwargs = {
        "env_name": env_cfg.env_name,
        "frame_skip": env_cfg.frame_skip,
        "from_pixels": env_cfg.from_pixels if from_pixels is None else from_pixels,  # for rendering
        "pixels_only": False,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.env_task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    return env


def make_transformed_env(base_env, env_cfg):
    if env_cfg.noop > 1:
        base_env = TransformedEnv(
            env=base_env,
            transform=NoopResetEnv(env_cfg.noop)
        )
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

    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))

    # obs_norm = ObservationNorm(in_keys=["pixels"])
    # env.append_transform(obs_norm)

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
    # init_stats(env, env_cfg.n_samples_stats, from_pixels)
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


def make_test_env(env_cfg):
    env = make_transformed_env(
        ParallelEnv(1, EnvCreator(lambda: make_base_env(env_cfg))), env_cfg
    )
    return env


# ====================================================================
# Collector and replay buffer
# ---------------------------

def make_collector(cfg, policy):
    env_cfg = cfg.env
    collector_cfg = cfg.collector
    # collector_class = MultiSyncDataCollector
    collector_class = SyncDataCollector
    state_dict = get_stats(env_cfg)
    collector = collector_class(
        # [make_parallel_env(env_cfg, state_dict=state_dict)] * collector_cfg.num_collectors,
        make_parallel_env(env_cfg, state_dict=state_dict),
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        # devices=collector_cfg.collector_devices,
        # init_random_frames=collector_cfg.init_random_frames,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
    )
    return collector


def make_data_buffer(cfg):
    cfg_collector = cfg.collector
    cfg_loss = cfg.loss
    sampler = SamplerWithoutReplacement()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg_collector.frames_per_batch), sampler=sampler,
        batch_size=cfg_loss.mini_batch_size,
    )


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.


def make_ppo_model(cfg):
    env_cfg = cfg.env
    model_cfg = cfg.model
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)
    # we must initialize the observation norm transform
    # init_stats(proof_environment, n_samples_stats=3, from_pixels=env_cfg.from_pixels)

    import gym
    import numpy as np
    import torch.nn as nn
    from pytorchrl.agent.actors.utils import init
    from pytorchrl.agent.actors.feature_extractors import CNN

    # 2.1 Define input keys
    in_keys = ["pixels"]
    obs_space = (4, 84, 84)  # obs_space.shape
    act_space = gym.spaces.Discrete(6)

    # 2.2 Define a shared Module and TensorDictModule (CNN + MLP)

    # Define shared net
    common_module = CNN(obs_space)
    dummy_obs = torch.zeros(8, *obs_space)
    common_module_output = common_module(dummy_obs)
    common_module = SafeModule(  # Like TensorDictModule
        module=common_module,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define one head for the policy
    init_ = lambda m: init(
        m,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        gain=np.sqrt(0.01))

    policy_net = init_(nn.Linear(common_module_output.shape[-1], act_space.n))
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        # out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        return_log_prob=True,
    )

    # Define one head for the value
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(0.01))
    value_net = init_(nn.Linear(common_module_output.shape[-1], 1))
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
        out_keys=["state_value"],
    )

    # 2.5 Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    # 2.6 Initialize the model by running a forward pass
    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    return actor, critic


def make_ppo_modules_state(model_cfg, proof_environment):
    actor_net, value_net = None, None
    return actor_net, value_net


def make_ppo_modules_pixels(model_cfg, proof_environment):
    actor_net, value_net = None, None
    return actor_net, value_net


def make_policy(policy_cfg):
    return make_ppo_model(policy_cfg)


# ====================================================================
# PPO Loss
# ---------


def make_advantage_module(loss_cfg, value_network):
    advantage_module = GAE(
        gamma=loss_cfg.gamma,
        lmbda=loss_cfg.gae_lamdda,
        value_network=value_network,
        average_gae=True,
    )
    return advantage_module


def make_loss(loss_cfg, actor_network, value_network):
    advantage_module = make_advantage_module(loss_cfg, value_network)
    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
        clip_epsilon=loss_cfg.clip_epsilon,
        loss_critic_type=loss_cfg.loss_critic_type,
        entropy_coef=loss_cfg.entropy_coef,
        critic_coef=loss_cfg.critic_coef,
        gamma=loss_cfg.gamma,
        normalize_advantage=True,
    )
    return loss_module, advantage_module


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
    exp_name = generate_exp_name("PPO", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(
        logger_cfg.backend, logger_name="ppo", experiment_name=exp_name
    )
    return logger


def make_recorder(cfg, logger, policy) -> Recorder:
    env_cfg = deepcopy(cfg.env)
    # env = make_transformed_env(make_base_env(env_cfg, from_pixels=True), env_cfg)
    env = make_transformed_env(
        ParallelEnv(1, EnvCreator(lambda: make_base_env(env_cfg))), env_cfg
    )
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
