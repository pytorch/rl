import pathlib
import warnings
from collections.abc import Callable
from functools import partial

import torch
from tensordict import TensorDict, TensorDictBase
from torch import optim

from torchrl.collectors import BaseCollector
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    OptimizationStepper,
    ReplayBufferTrainer,
    TargetNetUpdaterHook,
    Trainer,
    UpdateWeights,
    UTDRHook,
)


class TD3OptimizationStepper(OptimizationStepper):
    """Optimization stepper for TD3's multi-step update.

    Performs:
    1. Critic loss computation and backward pass
    2. Critic optimizer step
    3. Conditionally (every ``policy_update_delay`` steps): actor loss
       computation, backward pass, and actor optimizer step
    4. Replay-buffer priority update from TD error (when supported)

    Args:
        optimizer_actor (optim.Optimizer): Optimizer for the actor network.
        optimizer_critic (optim.Optimizer): Optimizer for the critic network.
        policy_update_delay (int): Actor is updated every this many steps.
        zero_grad_set_to_none (bool): Whether to pass ``set_to_none=True`` to
            ``optimizer.zero_grad()`` for both actor and critic optimizers.
            When ``True``, gradients are set to ``None`` instead of being zeroed.
        actor_loss_key (str): Key name for actor loss in the returned TensorDict.
        critic_loss_key (str): Key name for critic loss in the returned TensorDict.

    Note:
        If the trainer has a replay buffer exposing
        ``update_tensordict_priority``, priorities are updated immediately after
        the critic update using TD error from ``value_loss``.
    """

    def __init__(
        self,
        optimizer_actor: optim.Optimizer,
        optimizer_critic: optim.Optimizer,
        *,
        policy_update_delay: int = 2,
        zero_grad_set_to_none: bool = True,
        actor_loss_key: str = "loss_actor",
        critic_loss_key: str = "loss_qvalue",
    ) -> None:
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.policy_update_delay = int(policy_update_delay)
        self.zero_grad_set_to_none = zero_grad_set_to_none
        self.actor_loss_key = actor_loss_key
        self.critic_loss_key = critic_loss_key
        self._update_counter = 0

    def state_dict(self) -> dict:
        return {
            "update_counter": self._update_counter,
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._update_counter = int(state_dict.get("update_counter", 0))
        self.optimizer_actor.load_state_dict(state_dict["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state_dict["optimizer_critic"])

    @staticmethod
    def _params(optimizer: optim.Optimizer):
        for group in optimizer.param_groups:
            yield from group["params"]

    def step(
        self,
        trainer: Trainer,
        sub_batch: TensorDictBase,
    ) -> TensorDictBase:
        self._update_counter += 1
        do_actor = (self._update_counter % self.policy_update_delay) == 0

        clip_grad_norm = trainer.clip_grad_norm
        clip_norm = trainer.clip_norm

        q_loss, q_metadata = trainer.loss_module.value_loss(sub_batch)
        q_loss.backward()

        critic_params = list(self._params(self.optimizer_critic))
        if clip_grad_norm and clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(critic_params, clip_norm)
        elif clip_norm is not None:
            torch.nn.utils.clip_grad_value_(critic_params, clip_norm)

        self.optimizer_critic.step()
        self.optimizer_critic.zero_grad(set_to_none=self.zero_grad_set_to_none)

        actor_loss = q_loss.new_zeros(())
        if do_actor:
            actor_loss, *_ = trainer.loss_module.actor_loss(sub_batch)
            actor_loss.backward()

            actor_params = list(self._params(self.optimizer_actor))
            if clip_grad_norm and clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor_params, clip_norm)
            elif clip_norm is not None:
                torch.nn.utils.clip_grad_value_(actor_params, clip_norm)

            self.optimizer_actor.step()
            self.optimizer_actor.zero_grad(set_to_none=self.zero_grad_set_to_none)

        replay_buffer = getattr(trainer, "replay_buffer", None)
        if replay_buffer is not None and hasattr(
            replay_buffer, "update_tensordict_priority"
        ):
            priority_key = getattr(
                trainer.loss_module.tensor_keys, "priority", "td_error"
            )
            td_error = q_metadata["td_error"].detach().max(0)[0]
            sub_batch.set(priority_key, td_error)
            replay_buffer.update_tensordict_priority(sub_batch)

        return TensorDict(
            {
                self.critic_loss_key: q_loss.detach(),
                self.actor_loss_key: actor_loss.detach(),
            },
            batch_size=[],
        )


class TD3Trainer(Trainer):
    """A trainer class for Twin Delayed DDPG (TD3) algorithm.

    See also :class:`~torchrl.trainers.algorithms.configs.TD3TrainerConfig` for the
    Hydra configuration counterpart.

    This trainer implements the TD3 algorithm, an off-policy actor-critic method
    that builds on DDPG with improvements for stability including:
    - Clipped double Q-learning
    - Delayed policy updates
    - Target policy smoothing

    The trainer handles:
    - Replay buffer management for off-policy learning
    - Target network updates (typically SoftUpdate) for stable training
    - Policy weight updates to the data collector
    - Comprehensive logging of training metrics

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        frame_skip (int): Number of frames to skip between policy updates.
        optim_steps_per_batch (int): Number of optimization steps per collected batch.
        loss_module (LossModule | Callable): The TD3 loss module or a callable that computes losses.
        optimizer (optim.Optimizer, optional): Fallback optimizer for training. Defaults to None.
        optimization_stepper (TD3OptimizationStepper, optional): Custom optimization stepper
            controlling delayed actor/critic updates. Defaults to None.
        logger (Logger, optional): Logger for recording training metrics. Defaults to None.
        clip_grad_norm (bool, optional): Whether to clip gradient norms. Defaults to True.
        clip_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar during training. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_trainer_interval (int, optional): Interval for saving trainer state. Defaults to 10000.
        log_interval (int, optional): Interval for logging metrics. Defaults to 10000.
        save_trainer_file (str | pathlib.Path, optional): File path for saving trainer state. Defaults to None.
        num_epochs (int, optional): Number of epochs per batch. Defaults to 1 (typical for off-policy).
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing and sampling experiences. Defaults to None.
        enable_logging (bool, optional): Whether to enable metric logging. Defaults to True.
        log_rewards (bool, optional): Whether to log reward statistics. Defaults to True.
        log_actions (bool, optional): Whether to log action statistics. Defaults to True.
        log_observations (bool, optional): Whether to log observation statistics. Defaults to False.
        async_collection (bool, optional): Whether to use async collection. Defaults to False.
        log_timings (bool, optional): Whether to log timing information. Defaults to False.
        target_net_updater (TargetNetUpdater): Target network updater (typically SoftUpdate).
        exploration_module (torch.nn.Module, optional): Optional exploration module appended
            to actor weights when syncing policy parameters to the collector. Defaults to None.

    Note:
        This is an experimental/prototype feature. The API may change in future versions.
        TD3 is particularly effective for continuous control tasks.

    """

    def __init__(
        self,
        *,
        collector: BaseCollector,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        optimization_stepper: TD3OptimizationStepper | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        num_epochs: int = 1,
        replay_buffer: ReplayBuffer | None = None,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        async_collection: bool = False,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        target_net_updater: TargetNetUpdater,
        exploration_module: torch.nn.Module | None = None,
    ) -> None:
        warnings.warn(
            "TD3Trainer is an experimental/prototype feature. The API may change in future versions. "
            "Please report any issues or feedback to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            num_epochs=num_epochs,
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        if replay_buffer is not None:
            rb_trainer = ReplayBufferTrainer(
                replay_buffer,
                batch_size=None,
                flatten_tensordicts=True,
                memmap=False,
                device=getattr(replay_buffer.storage, "device", "cpu"),
                iterate=True,
            )
            if not self.async_collection:
                self.register_op("pre_epoch", rb_trainer.extend)
            self.register_op("process_optim_batch", rb_trainer.sample)
        # Note: the replay buffer priorities are updated as part of the optimization
        # stepper, as this step requires access to the critic loss.

        self.register_op("post_optim", TargetNetUpdaterHook(target_net_updater))

        # Here, we build a weight source that mirrors the collector policy structure when
        # an exploration module is used, to allow for simpler weight synchronization.
        weights_source = self.loss_module.actor_network
        if exploration_module is not None:
            from tensordict.nn import TensorDictSequential

            weights_source = TensorDictSequential(weights_source, exploration_module)

        policy_weights_getter = partial(TensorDict.from_module, weights_source)
        update_weights = UpdateWeights(
            self.collector, 1, policy_weights_getter=policy_weights_getter
        )
        self.register_op("post_steps", update_weights)

        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_actions = log_actions
        self.log_observations = log_observations

        if self.enable_logging:
            self._setup_td3_logging()

    def _setup_td3_logging(self):
        """Set up logging hooks for TD3-specific metrics."""
        hook_dest = "pre_steps_log" if not self.async_collection else "post_optim_log"

        log_done_percentage = LogScalar(
            key=("next", "done"),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,
            reduction="mean",
        )
        self.register_op(hook_dest, log_done_percentage)

        if self.log_rewards:
            log_rewards = LogScalar(
                key=("next", "reward"),
                logname="r_training",
                log_pbar=True,
                include_std=True,
                reduction="mean",
            )
            log_max_reward = LogScalar(
                key=("next", "reward"),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            log_total_reward = LogScalar(
                key=("next", "reward"),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="sum",
            )
            self.register_op(hook_dest, log_rewards)
            self.register_op(hook_dest, log_max_reward)
            self.register_op(hook_dest, log_total_reward)

        if self.log_actions:
            log_action_norm = LogScalar(
                key="action",
                logname="action_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_action_norm)

        if self.log_observations:
            log_obs_norm = LogScalar(
                key="observation",
                logname="obs_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_obs_norm)

        self.register_op("pre_steps_log", UTDRHook(self))
