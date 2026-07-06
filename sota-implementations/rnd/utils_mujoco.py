# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.nn as nn

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RNDTransform
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(
    env_name="HalfCheetah-v4",
    device="cpu",
    from_pixels: bool = False,
    rnd_transform: RNDTransform | None = None,
):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    if rnd_transform is not None:
        env.append_transform(rnd_transform)
    return env


# ====================================================================
# RND network utils
# --------------------------------------------------------------------


def make_rnd_networks(obs_dim: int, embed_dim: int = 64, device="cpu"):
    """Create the frozen target and trainable predictor networks for RND.

    Both networks map observations to a fixed-dimensional embedding. The target
    is randomly initialised and permanently frozen; the predictor is trained to
    match it, producing lower error on familiar states and higher error on novel
    ones — which becomes the intrinsic reward.

    Args:
        obs_dim: dimensionality of the (already normalised) observation.
        embed_dim: embedding dimension for both networks. The predictor uses an
            extra hidden layer so it has strictly more capacity than the target,
            following the original paper.
        device: device to place the networks on.

    Returns:
        target (nn.Module): frozen random network.
        predictor (nn.Module): trainable network.
    """
    target = nn.Sequential(
        nn.Linear(obs_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    ).to(device)

    # Predictor has an extra hidden layer so it has strictly more capacity
    # than the target, following the original paper's architecture.
    predictor = nn.Sequential(
        nn.Linear(obs_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    ).to(device)

    # Orthogonal init for both — keeps embedding norms stable at the start.
    for net in (target, predictor):
        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                layer.bias.data.zero_()

    target.requires_grad_(False)
    return target, predictor


# ====================================================================
# PPO model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):
    input_shape = proof_environment.observation_spec["observation"].shape
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=nn.Tanh,
        out_features=num_outputs,
        num_cells=[64, 64],
        device=device,
    )
    for layer in policy_mlp.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()
    policy_mlp = nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )
    for layer in value_mlp.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()
    value_module = ValueOperator(value_mlp, in_keys=["observation"])

    return policy_module, value_module


def make_ppo_models(env_name, device):
    proof_environment = make_env(env_name, device=device)
    actor, critic = make_ppo_models_state(proof_environment, device=device)
    proof_environment.close()
    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        test_env.apply(dump_video)
    del td_test
    return torch.cat(test_rewards, 0).mean()
