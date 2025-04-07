# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Template for a PPO model on the pendulum env, for Lightning."""

import typing as ty

import torch
from tensordict import TensorDict  # type: ignore

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvBase,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import MLP

from ._base import BaseRL


class PPO(BaseRL):
    """Basic PPO Model.

    See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        **kwargs: ty.Any,
    ):
        kwargs.setdefault("model", "ppo")
        super().__init__(**kwargs)

    def advantage(self, batch: TensorDict) -> None:
        """We'll need an "advantage" signal to make PPO work.

        We re-compute it at each epoch as its value depends on the value
        network which is updated in the inner loop.
        """
        with torch.no_grad():
            try:
                assert self.advantage_module is not None
                self.advantage_module(batch)
            except RuntimeError as ex:
                raise RuntimeError(f"{ex}\n{batch}") from ex


class PPOPendulum(PPO):
    """Basic PPO Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        num_cells: int = 256,
        n_mlp_layers: int = 3,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            env (ty.Union[str, EnvBase], optional):
                Environment name.
                Defaults to "InvertedDoublePendulum-v4".
            num_cells (int, optional): _description_. Defaults to 256.
            lr (float, optional): _description_. Defaults to 3e-4.
            max_grad_norm (float, optional): _description_. Defaults to 1.0.
            frame_skip (int, optional): _description_. Defaults to 1.
            frames_per_batch (int, optional): _description_. Defaults to 100.
            total_frames (int, optional): _description_. Defaults to 100_000.
            accelerator (ty.Union[str, torch.device], optional): _description_. Defaults to "cpu".
            sub_batch_size (int, optional):
                Cardinality of the sub-samples gathered from the current data in the inner loop.
                Defaults to `1`.
            clip_epsilon (float, optional): _description_. Defaults to 0.2.
            gamma (float, optional): _description_. Defaults to 0.99.
            lmbda (float, optional): _description_. Defaults to 0.95.
            entropy_eps (float, optional): _description_. Defaults to 1e-4.
            lr_monitor (str, optional): _description_. Defaults to "loss/train".
            lr_monitor_strict (bool, optional): _description_. Defaults to False.
            rollout_max_steps (int, optional): _description_. Defaults to 1000.
            n_mlp_layers (int, optional): _description_. Defaults to 3.
            in_keys (ty.List[str], optional): _description_. Defaults to ["observation"].
            flatten (bool, optional): _description_. Defaults to False.
            flatten_start_dim (int, optional): _description_. Defaults to 0.
            legacy (bool, optional): _description_. Defaults to False.
            automatic_optimization (bool, optional): _description_. Defaults to True.
        """
        self.env_name = "InvertedDoublePendulum-v4"
        self.device_info = kwargs.get("device", "cpu")
        self.frame_skip = kwargs.get("frame_skip", 1)
        base_env = self.make_env()
        out_features = base_env.action_spec.shape[-1]
        actor_nn = MLP(
            out_features=2 * out_features,
            depth=n_mlp_layers,
            num_cells=num_cells,
            dropout=True,
        )
        value_nn = MLP(
            out_features=1,
            depth=n_mlp_layers,
            num_cells=num_cells,
            dropout=True,
        )
        # Call superclass
        super().__init__(
            env_name="InvertedDoublePendulum-v4",
            actor_nn=actor_nn,
            value_nn=value_nn,
            **kwargs,
        )

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        obs_norm = ObservationNorm(in_keys=["observation"])
        double2float = DoubleToFloat()
        env = TransformedEnv(
            base_env,
            transform=Compose(
                obs_norm,
                double2float,
                StepCounter(),
            ),
        )
        obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        return env
