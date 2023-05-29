import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss


# ====================================================================
# Environment utils
# -----------------


def env_maker(task, frame_skip=1, device="cpu", from_pixels=False):
    return GymEnv(task, device=device, frame_skip=frame_skip, from_pixels=from_pixels)


def apply_env_transforms(env, reward_scaling=1.0):
    transformed_env = TransformedEnv(
        env,
        Compose(
            RewardScaling(loc=0.0, scale=reward_scaling),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
        ),
    )
    return transformed_env


def make_environment(cfg):
    """Make environments for training and evaluation."""
    parallel_env = ParallelEnv(
        cfg.env_per_collector, EnvCreator(lambda: env_maker(task=cfg.env_name))
    )
    parallel_env.set_seed(cfg.seed)

    train_env = apply_env_transforms(parallel_env)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.env_per_collector, EnvCreator(lambda: env_maker(task=cfg.env_name))
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = MultiSyncDataCollector(
        # we'll just run one ParallelEnvironment. Adding elements to the list would increase the number of envs run in parallel
        [
            train_env,
        ],
        actor_model_explore,
        frames_per_batch=cfg.frames_per_batch,
        max_frames_per_traj=cfg.max_frames_per_traj,
        total_frames=cfg.total_frames,
        device=cfg.collector_device,
    )
    collector.set_seed(cfg.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir="/tmp/",
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.default_policy_scale}",
        scale_lb=cfg.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.loss_function,
        delay_actor=False,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, tau=cfg.target_update_polyak)
    return loss_module, target_net_updater


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(actor_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer_critic = optim.Adam(
        critic_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    optimizer_alpha = optim.Adam([loss_module.log_alpha], lr=cfg.lr)
    return optimizer_actor, optimizer_critic, optimizer_alpha
