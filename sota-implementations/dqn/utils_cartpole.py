# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
import torch.nn
import torch.optim
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import RewardSum, StepCounter, Transform, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, QValueActor
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="CartPole-v1", device="cpu", from_pixels=False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def make_render_env(spec: Any):
    """Builds a CartPole environment suitable for ``rlrender``.

    The Gymnasium classic-control pixel renderer requires the optional
    ``pygame`` dependency. To keep the SOTA script renderable in a minimal
    TorchRL checkout, this factory draws lightweight RGB frames from the
    CartPole state when ``rlrender --from-pixels`` is requested.
    """
    checkpoint = (
        spec.checkpoint if isinstance(getattr(spec, "checkpoint", None), dict) else {}
    )
    env_name = spec.env_kwargs.get(
        "env_name", checkpoint.get("env_name", "CartPole-v1")
    )
    env = make_env(env_name, device=spec.device, from_pixels=False)
    if spec.from_pixels:
        env.append_transform(_CartPolePixelTransform())
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
        device=device,
    )

    qvalue_module = QValueActor(
        module=mlp,
        spec=Composite(action=action_spec).to(device),
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(env_name, device):
    proof_environment = make_env(env_name, device=device)
    qvalue_module = make_dqn_modules(proof_environment, device=device)
    del proof_environment
    return qvalue_module


def make_render_policy(spec: Any):
    """Builds the DQN policy module for ``rlrender`` checkpoint loading."""
    checkpoint = spec.checkpoint if isinstance(spec.checkpoint, dict) else {}
    env_name = spec.policy_kwargs.get(
        "env_name", checkpoint.get("env_name", "CartPole-v1")
    )
    return make_dqn_model(env_name, device=spec.device)


class _CartPolePixelTransform(Transform):
    def __init__(self, *, height: int = 240, width: int = 320) -> None:
        super().__init__(in_keys=["observation"], out_keys=["pixels"])
        self.height = height
        self.width = width

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        return _cartpole_pixels(observation, height=self.height, width=self.width)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec = observation_spec.clone()
        observation_spec["pixels"] = Unbounded(
            shape=(self.height, self.width, 3),
            dtype=torch.uint8,
            device=observation_spec.device,
        )
        return observation_spec


_CARTPOLE_BACKGROUND = torch.tensor([238, 242, 248], dtype=torch.uint8)
_CARTPOLE_TRACK_COLOR = torch.tensor([90, 100, 120], dtype=torch.uint8)
_CARTPOLE_CART_COLOR = torch.tensor([40, 90, 180], dtype=torch.uint8)
_CARTPOLE_POLE_COLOR = torch.tensor([220, 90, 30], dtype=torch.uint8)
_CARTPOLE_AXLE_COLOR = torch.tensor([30, 30, 30], dtype=torch.uint8)


def _cartpole_pixels(
    observation: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    source_device = observation.device
    flat_observation = observation.detach().cpu().reshape(-1, observation.shape[-1])
    frames = torch.empty(flat_observation.shape[0], height, width, 3, dtype=torch.uint8)
    frames[:] = _CARTPOLE_BACKGROUND
    track_y = int(0.74 * height)
    frames[:, track_y : track_y + 2, :, :] = _CARTPOLE_TRACK_COLOR
    world_width = 4.8
    scale = width / world_width
    cart_width = max(8, int(0.50 * scale))
    cart_height = max(6, int(0.30 * scale))
    pole_length = int(1.00 * scale)
    for index, row in enumerate(flat_observation):
        cart_position = float(row[0])
        cart_x = int(width / 2 + cart_position * scale)
        cart_top = track_y - cart_height // 2
        cart_bottom = min(track_y + cart_height // 2, height - 1)
        cart_left = max(cart_x - cart_width // 2, 0)
        cart_right = min(cart_x + cart_width // 2, width - 1)
        frames[index, cart_top:cart_bottom, cart_left:cart_right] = _CARTPOLE_CART_COLOR
        pole_x = int(round(cart_x + pole_length * torch.sin(row[2]).item()))
        pole_y = int(round(track_y - pole_length * torch.cos(row[2]).item()))
        _draw_line(
            frames[index],
            cart_x,
            track_y - cart_height // 2,
            pole_x,
            pole_y,
            color=_CARTPOLE_POLE_COLOR,
            radius=3,
        )
        _draw_disk(
            frames[index], cart_x, track_y - cart_height // 2, 5, _CARTPOLE_AXLE_COLOR
        )
    return frames.reshape(*observation.shape[:-1], height, width, 3).to(source_device)


def _draw_line(
    frame: torch.Tensor,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    color: torch.Tensor,
    radius: int,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    alphas = torch.linspace(0.0, 1.0, steps + 1)
    xs = ((1 - alphas) * x0 + alphas * x1).round().long()
    ys = ((1 - alphas) * y0 + alphas * y1).round().long()
    height, width = frame.shape[:2]
    min_x = max(int(xs.min()) - radius, 0)
    max_x = min(int(xs.max()) + radius + 1, width)
    min_y = max(int(ys.min()) - radius, 0)
    max_y = min(int(ys.max()) + radius + 1, height)
    if min_x >= max_x or min_y >= max_y:
        return
    grid_y = torch.arange(min_y, max_y).view(-1, 1, 1)
    grid_x = torch.arange(min_x, max_x).view(1, -1, 1)
    distance_sq = (grid_x - xs.view(1, 1, -1)) ** 2 + (grid_y - ys.view(1, 1, -1)) ** 2
    mask = (distance_sq <= radius**2).any(-1)
    frame[min_y:max_y, min_x:max_x][mask] = color


def _draw_disk(
    frame: torch.Tensor,
    cx: int,
    cy: int,
    radius: int,
    color: torch.Tensor,
) -> None:
    height, width = frame.shape[:2]
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
    if y0 >= y1 or x0 >= x1:
        return
    ys = torch.arange(y0, y1).unsqueeze(1)
    xs = torch.arange(x0, x1).unsqueeze(0)
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    frame[y0:y1, x0:x1][mask] = color


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
