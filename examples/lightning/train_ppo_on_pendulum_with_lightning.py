# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.envs import EnvBase, GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.trainers import RLTrainingLoop


def make_env(env_name: str = "InvertedDoublePendulum-v4", **kwargs: ty.Any) -> EnvBase:
    """Utility function to init an env."""
    env = GymEnv(env_name, **kwargs)
    return env


def create_nets(base_env: EnvBase) -> ty.Tuple[torch.nn.Module, torch.nn.Module]:
    out_features = base_env.action_spec.shape[-1]
    actor_nn = MLP(
        out_features=2 * out_features,
        depth=3,
        num_cells=256,
        dropout=True,
    )
    value_nn = MLP(
        out_features=1,
        depth=3,
        num_cells=256,
        dropout=True,
    )
    return actor_nn, value_nn


def make_actor(env: EnvBase, actor_nn: torch.nn.Module) -> ProbabilisticActor:
    # Actor
    actor_net = torch.nn.Sequential(
        actor_nn,
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    ).double()
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
    return policy_module


def make_critic(
    env: EnvBase,
    value_nn: torch.nn.Module,
    gamma: float = 0.99,
    lmbda: float = 0.95,
) -> ty.Tuple[ValueOperator, GAE]:
    value_module = ValueOperator(
        module=value_nn,
        in_keys=["observation"],
    ).double()
    td = env.reset()
    value_module(td)
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
    )
    return value_module, advantage_module


def make_loss_module(
    policy_module,
    value_module,
    advantage_module,
    entropy_eps: float = 1e-4,
    clip_epsilon: float = 0.2,
    loss_function: str = "smooth_l1",
) -> ClipPPOLoss:
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type=loss_function,
    )
    loss_module.set_keys(value_target=advantage_module.value_target_key)
    return loss_module


class PPOPendulum(RLTrainingLoop):
    def make_env(self) -> EnvBase:
        """You have to implement this method, which has to take no inputs and return
        your environment."""
        return make_env()


def create_model() -> pl.LightningModule:
    env = make_env()
    actorn_nn, critic_nn = create_nets(env)
    policy_module = make_actor(env, actorn_nn)
    value_module, advantage_module = make_critic(env, critic_nn)
    loss_module = make_loss_module(policy_module, value_module, advantage_module)
    frame_skip = 1
    frames_per_batch = frame_skip * 5
    total_frames = 100
    model = PPOPendulum(
        loss_module=loss_module,
        policy_module=policy_module,
        advantage_module=advantage_module,
        frame_skip=frame_skip,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        use_checkpoint_callback=True,
    )
    return model


def main() -> None:
    model = create_model()
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=4,
        val_check_interval=2,
        log_every_n_steps=1,
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
