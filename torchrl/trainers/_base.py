# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Creates a helper class for more complex models."""

import typing as ty

import torch
from tensordict.nn import TensorDictModule  # type: ignore
from tensordict.nn.distributions import NormalParamExtractor  # type: ignore

from torchrl.envs import EnvBase, GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, CQLLoss, SoftUpdate
from torchrl.objectives.value import GAE

from .loops import RLTrainingLoop


class BaseRL(RLTrainingLoop):
    """Base for RL Model. See: https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop"""

    def __init__(
        self,
        actor_nn: torch.nn.Module,
        value_nn: torch.nn.Module,
        env_name: str = "InvertedDoublePendulum-v4",
        model: str = "ppo",
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        clip_epsilon: float = 0.2,
        alpha_init: float = 1,
        loss_function: str = "smooth_l1",
        flatten_state: bool = False,
        tau: float = 1e-2,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            env (ty.Union[str, EnvBase], optional): _description_. Defaults to "InvertedDoublePendulum-v4".
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
            flatten (bool, optional): _description_. Defaults to False.
            flatten_start_dim (int, optional): _description_. Defaults to 0.
            legacy (bool, optional): _description_. Defaults to False.
            automatic_optimization (bool, optional): _description_. Defaults to True.
        """
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
                "actor_nn",
                "value_nn",
            ]
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.env_name = env_name
        self.device_info = kwargs.get("device", "cpu")
        self.frame_skip = kwargs.get("frame_skip", 1)
        # Environment
        base_env = self.make_env()
        # Env transformations
        env = self.transformed_env(base_env)
        # Actor
        actor_net = torch.nn.Sequential(
            torch.nn.Flatten(0) if flatten_state else torch.nn.Identity(),
            actor_nn,
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        td = env.reset()
        policy_module(td)
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": 0,  # env.action_spec.space.minimum,
                "max": 1,  # env.action_spec.space.maximum,
            },
            return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
        )
        # Critic and loss depend on the model
        target_net_updater = None
        if model in ["cql"]:
            advantage_module = None
            # Q-Value
            value_module = ValueOperator(
                module=value_nn,
                in_keys=["observation", "action"],
                out_keys=["state_action_value"],
            )
            td = env.reset()
            td = env.rand_action(td)
            td = env.step(td)
            td = value_module(td)
            # Loss CQL
            loss_module = CQLLoss(
                actor_network=policy_module,
                qvalue_network=value_module,
                action_spec=env.action_spec,
                alpha_init=alpha_init,
                loss_function=loss_function,
            )
            loss_module.make_value_estimator(gamma=gamma)
            target_net_updater = SoftUpdate(loss_module, tau=tau)
        elif model in ["ppo"]:
            # Value
            value_net = torch.nn.Sequential(
                torch.nn.Flatten(1) if flatten_state else torch.nn.Identity(),
                value_nn,
            )
            value_module = ValueOperator(
                module=value_net,
                in_keys=["observation"],
            )
            td = env.reset()
            value_module(td)
            # Loss PPO
            advantage_module = GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=value_module,
                average_gae=True,
            )
            loss_module = ClipPPOLoss(
                actor=policy_module,
                critic=value_module,
                clip_epsilon=clip_epsilon,
                entropy_bonus=bool(entropy_eps),
                entropy_coef=entropy_eps,
                # these keys match by default but we set this for completeness
                critic_coef=1.0,
                # gamma=0.99,
                loss_critic_type=loss_function,
            )
            loss_module.set_keys(value_target=advantage_module.value_target_key)
        else:
            raise ValueError(f"Unrecognized model {model}")
        # Call superclass
        super().__init__(
            loss_module=loss_module,
            policy_module=policy_module,
            advantage_module=advantage_module,
            target_net_updater=target_net_updater,
            **kwargs,
        )

    def make_env(self) -> EnvBase:
        """Utility function to init an env.

        Args:
            env (ty.Union[str, EnvBase]): _description_

        Returns:
            EnvBase: _description_
        """
        env = GymEnv(
            self.env_name,
            device=self.device_info,
            frame_skip=self.frame_skip,
        )
        return env
