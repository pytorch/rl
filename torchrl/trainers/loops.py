# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=arguments-differ
"""This module creates a `RLTrainingLoop` base class for integration with `lightning`."""


import typing as ty

try:
    import lightning.pytorch as pl
    import lightning.pytorch.callbacks as cb
    from lightning.fabric.utilities.types import LRScheduler
    from lightning.pytorch.core.optimizer import LightningOptimizer
    from lightning.pytorch.utilities.types import (
        LRSchedulerConfigType,
        OptimizerLRSchedulerConfig,
    )

    HAS_PL = True
except ImportError:
    HAS_PL = False

import torch
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule  # type: ignore
from torch import Tensor

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import EnvBase, EnvCreator, ParallelEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate

from .collector import CollectorDataset


class RLTrainingLoop(pl.LightningModule):
    """RL training loop.

    This class inherits from PyTorch Lightning `LightningModule` and makes use
    of its API and hooks to correcly recreate the training loop shown at:
    https://pytorch.org/rl/tutorials/coding_ppo.html#training-loop

    By populating the correct hooks with the right code from the RL training loop,
    one can create a new RL model by inheriting from this class, and their model
    can then be trained using PyTorch Lightning's `Trainer`.
    """

    def __init__(
        self,
        loss_module: TensorDictModule,
        policy_module: TensorDictModule,
        advantage_module: TensorDictModule = None,
        target_net_updater: SoftUpdate | None = None,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        frame_skip: int = 1,
        frames_per_batch: int = 100,
        total_frames: int = 100_000,
        sub_batch_size: int = 1,
        lr_monitor: str = "loss/train",
        lr_monitor_strict: bool = False,
        rollout_max_steps: int = 1000,
        automatic_optimization: bool = True,
        use_checkpoint_callback: bool = False,
        save_every_n_train_steps: int = 100,
        raise_error_on_nan: bool = False,
        num_envs: int = 1,
        env_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
    ) -> None:
        """
        Args:
            loss_module (TensorDictModule):
                A torchl loss module.

            policy_module (TensorDictModule):
                Policy module.

            advantage_module (TensorDictModule, optional):
                Value module.

            target_net_updater (SoftUpdate, optional):
                Target network updater. Defaults to None.

            lr (float, optional):
                Learning rate. Defaults to 3e-4.

            max_grad_norm (float, optional):
                Maximun norm value for the gradients. Defaults to 1.0.

            frame_skip (int, optional):
                Frame skip value. Defaults to 1.

            frames_per_batch (int, optional):
                Number of frames per batch. Defaults to 100.

            total_frames (int, optional):
                Total number of frames. Defaults to 100_000.

            sub_batch_size (int, optional):
                Sub batch size. Defaults to 1.

            lr_monitor (str, optional):
                Name given to the loss to monitor.
                Defaults to "loss/train".

            lr_monitor_strict (bool, optional):
                Whetehr to raise an error in case the metric `lr_monitor` is not found.
                Defaults to False.

            rollout_max_steps (int, optional):
                Max number of rollout steps.
                Defaults to 1000.

            automatic_optimization (bool, optional):
                Whether to use manual or automatic optimization.
                Defaults to True.

            use_checkpoint_callback (bool, optional):
                Whether to use a checkpoint callback.
                Defaults to False.

            save_every_n_train_steps (int, optional):
                Save every n steps. Defaults to 100.

            raise_error_on_nan (bool, optional):
                Whether to immediately raise an error if a NaN is encountered.
                Defaults to False.

            num_envs (int, optional):
                Number of environments to use.
                If `num_envs > 1`, `num_envs` copies of the input env will be created.
                Defaults to 1.

            env_kwargs (ty.Dict[str, ty.Any], optional):
                Parameters for your environment.
                Defaults to {}.
        """
        if not HAS_PL:
            raise RuntimeError(
                "PyTorch Lightning is not installed. Please run `pip install lightning`."
            )
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "base_env",
                "env",
                "loss_module",
                "policy_module",
                "value_module",
                "advantage_module",
                "target_net_updater",
            ]
        )
        if not hasattr(self, "env_kwargs"):
            self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.raise_error_on_nan = raise_error_on_nan
        self.use_checkpoint_callback = use_checkpoint_callback
        self.save_every_n_train_steps = save_every_n_train_steps
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.frame_skip = frame_skip
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.rollout_max_steps = rollout_max_steps
        self.num_envs = num_envs
        # Environment
        self.env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=EnvCreator(self._make_env),
            serial_for_single=True,
        )
        # Modules
        self.loss_module = loss_module
        self.policy_module = policy_module
        self.advantage_module = advantage_module
        self.target_net_updater = target_net_updater
        # Important: This property activates manual optimization
        self.automatic_optimization = automatic_optimization
        # Will exist only after training initialisation
        self.optimizer: torch.optim.Adam
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
        self.lr_monitor = lr_monitor
        self.lr_monitor_strict = lr_monitor_strict
        self._dataset: CollectorDataset

    @property
    def replay_buffer(self) -> ReplayBuffer:
        """Gets replay buffer from collector."""
        return self.dataset.replay_buffer

    @property
    def dataset(self) -> CollectorDataset:
        """Gets dataset."""
        _dataset = getattr(self, "_dataset", None)
        if not isinstance(_dataset, CollectorDataset):
            self._dataset = CollectorDataset(
                env=self.env,
                policy_module=self.policy_module,
                frames_per_batch=self.frames_per_batch,
                total_frames=self.total_frames,
                device=self.device,
                # batch_size=self.sub_batch_size,
            )
        return self._dataset

    def setup(self, stage: ty.Optional[str] = None) -> None:
        """Set up."""

    def train_dataloader(self) -> ty.Iterable[TensorDict]:
        """Create DataLoader for training."""
        self._dataset = None  # type: ignore
        return self.dataset

    def configure_callbacks(self) -> ty.Sequence[pl.Callback]:
        """Configure checkpoint."""
        callbacks = []
        if self.use_checkpoint_callback:
            ckpt_cb = cb.ModelCheckpoint(
                monitor="loss/train",
                mode="min",
                save_top_k=3,
                save_last=True,
                save_on_train_epoch_end=True,
                every_n_train_steps=self.save_every_n_train_steps,
            )
            callbacks.append(ckpt_cb)
        return callbacks

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer (`torch.optim.Adam`) and the learning rate
        scheduler (`torch.optim.lr_scheduler.CosineAnnealingLR`)."""
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.lr)
        try:
            max_steps = self.trainer.max_steps
        except RuntimeError:
            max_steps = 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            max(1, max_steps // self.frames_per_batch),
            0.0,
        )
        lr_scheduler = LRSchedulerConfigType(  # type: ignore
            scheduler=self.scheduler,
            monitor=self.lr_monitor,
            strict=self.lr_monitor_strict,
        )
        cfg = OptimizerLRSchedulerConfig(
            optimizer=self.optimizer, lr_scheduler=lr_scheduler
        )
        return cfg

    def on_validation_epoch_start(self) -> None:
        """Validation step."""
        self.rollout()

    def on_test_epoch_start(self) -> None:
        """Test step."""
        self.rollout()

    def training_step(
        self,
        batch: TensorDict,
        batch_idx: int,
    ) -> Tensor:
        """Implementation follows the PyTorch tutorial:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"""
        # Run optimization step
        loss = self.step(batch, batch_idx=batch_idx, tag="train")
        # This will run only if manual optimization
        self.manual_optimization_step(loss)
        # Update target network
        if isinstance(self.target_net_updater, SoftUpdate):
            self.target_net_updater.step()
        # We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data
        n = self.trainer.val_check_interval
        if n is None:
            n = 10  # pragma: no cover
        n = int(n)
        if batch_idx % n == 0:
            self.rollout()
        # Return loss
        return loss

    def on_train_batch_end(
        self,
        outputs: Tensor | ty.Mapping[str, ty.Any] | None,
        batch: ty.Any,
        batch_idx: int,
    ) -> None:
        """Check if we have to stop.
        For some reason, Lightning can't understand this.
        Probably because we are using an `IterableDataset`."""
        # Stop on max steps
        global_step = self.trainer.global_step
        max_steps = self.trainer.max_steps
        if global_step >= max_steps:
            self.stop(f"global_step={global_step} > max_steps={max_steps}")
            return
        # Stop on max epochs
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        if (
            isinstance(max_epochs, int)
            and max_epochs > 0
            and current_epoch >= max_epochs
        ):
            self.stop(f"current_epoch={current_epoch} > max_epochs={max_epochs}")
            return
        # Stop on total frames
        if global_step >= self.total_frames:
            self.stop(f"global_step={global_step} > total_frames={self.total_frames}")
            return

    def stop(self, msg: str = "") -> None:
        """Change `Trainer` flag to make this stop."""
        self.trainer.should_stop = True

    def manual_optimization_step(self, loss: Tensor) -> None:
        """Steps to run if manual optimization is enabled."""
        if self.automatic_optimization:
            return
        # Get optimizers
        optimizer = self.optimizers()
        if not isinstance(optimizer, (torch.optim.Optimizer, LightningOptimizer)):
            raise TypeError(
                f"Method `self.optimizers` returned a {type(optimizer)} object. "
                "Please make sure that the method `self.configure_optimizers()` "
                "returns the expected output as explained here: "
                "https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers"
            )
        # Zero grad before accumulating them
        optimizer.zero_grad()
        # Run backward
        self.manual_backward(loss)
        # Clip gradients if necessary
        self.clip_gradients()
        # Optimizer
        optimizer.step()
        # Call schedulers
        self.call_scheduler()

    def clip_gradients(  # type: ignore
        self,
        optimizer: ty.Optional[torch.optim.Optimizer] = None,
        gradient_clip_val: ty.Optional[ty.Union[int, float]] = None,
        gradient_clip_algorithm: ty.Optional[str] = None,
    ) -> None:
        """Clip gradients if necessary. This is an official hook."""
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            clip_val = self.max_grad_norm
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), clip_val)

    def call_scheduler(self) -> None:
        """Call schedulers. We are using an infinite datalaoder,
        this will never be called by the `pl.Trainer` in the `on_train_epoch_end` hook.
        We have to call it manually in the `training_step`."""
        scheduler = self.lr_schedulers()
        if not isinstance(scheduler, LRScheduler):
            raise TypeError(
                f"Method `self.lr_schedulers` returned a {type(scheduler)} object. "
                "Please make sure that the method `self.configure_optimizers()` "
                "returns the expected output as explained here: "
                "https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers"
            )
        try:
            # c = self.trainer.callback_metrics[self.lr_monitor]
            scheduler.step(self.trainer.global_step)
        except Exception as ex:  # pylint: disable=broad-exception-caught
            pass

    def advantage(self, batch: TensorDict) -> None:
        """Advantage step.

        Some models (like PPO) need an advantage signal.
        They can implement this method to do that.

        For example:
        ```python
        def advantage(self, batch: TensorDict) -> None:
            with torch.no_grad():
                self.advantage_module(batch)
        ```
        """

    def step(
        self,
        batch: TensorDict,
        batch_idx: int = 0,
        tag: str = "train",
    ) -> Tensor:
        """This method will possible call the advantage module (if any),
        then sample from the replay buffer and compute the loss.

        Args:
            batch (TensorDict):
                Batch object returned by the data loader.

            batch_idx (int, optional):
                Batch ID. Defaults to 0.

            tag (str, optional):
                "train", "val" or "test". Defaults to "train".

        Raises:
            RuntimeError: If `self.frames_per_batch // self.sub_batch_size < 1`.

        Returns:
            Tensor: Loss.
        """
        # Call advantage hook: this can also be an empty method
        self.advantage(batch)
        # Initialize loss
        loss = torch.tensor(0.0).to(self.device)
        # Sanity check
        n: int = self.frames_per_batch // self.sub_batch_size
        if n < 1:
            raise RuntimeError(
                f"frames_per_batch({self.frames_per_batch}) // "
                f"sub_batch_size({self.sub_batch_size}) = {n} should be > {0}."
            )
        # Evaluate and accumulate loss
        for _ in range(n):
            subdata: TensorDict = self.replay_buffer.sample(self.sub_batch_size)
            loss_vals: TensorDict = self.loss(subdata.to(self.device))
            loss, losses = self.collect_loss(loss_vals, loss, tag)
        # Log stuff
        self.log_dict(losses)
        self.log(f"loss/{tag}", loss, prog_bar=True)
        reward: Tensor = batch["next", "reward"]
        self.log(f"reward/{tag}", reward.mean().item(), prog_bar=True)
        if "step_count" in batch:
            step_count: Tensor = batch["step_count"]
            self.log(f"step_count/{tag}", step_count.max().item(), prog_bar=True)
        # Return loss value
        return loss

    def loss(self, data: TensorDict) -> TensorDict:
        """Evaluates the loss over input data."""
        loss_vals: TensorDict = self.loss_module(data.to(self.device))
        return loss_vals

    def collect_loss(
        self,
        loss_vals: TensorDict,
        loss: ty.Optional[torch.Tensor] = None,
        tag: str = "train",
    ) -> ty.Tuple[torch.Tensor, ty.Dict[str, torch.Tensor]]:
        """Updates the input loss and extracts losses from input `TensorDict`
        and collects them into a dict."""
        # Initialize loss
        if loss is None:
            loss = torch.tensor(0.0).to(loss_vals.device)
        # Initialize output
        loss_dict: ty.Dict[str, torch.Tensor] = {}
        # Iterate over losses
        for key, value in loss_vals.items():
            # Loss actually have a key that starts with "loss_"
            if "loss_" in key:
                # If not finite, may raise error
                if self.raise_error_on_nan:
                    if value.isnan().any() or value.isinf().any():
                        raise RuntimeError(f"Invalid loss value for {key}: {value}.")
                # Update total loss
                loss = loss + value
                loss_dict[f"{key}/{tag}"] = value
        # Sanity check and return
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"Loss expected to be {torch.Tensor} but found of type {type(loss)}."
            )
        return loss, loss_dict

    def rollout(self, tag: str = "eval") -> None:
        """We evaluate the policy once every `sefl.trainer.val_check_interval` batches of data.

        Evaluation is rather simple:
        execute the policy without exploration (take the expected value of the action distribution)
        for a given number of steps.

        The `self.env.rollout()` method can take a policy as argument:
        it will then execute this policy at each step.
        """
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = self.env.rollout(self.rollout_max_steps, self.policy_module)
            reward = eval_rollout["next", "reward"]
            self.log(f"reward/{tag}", reward.mean().item())
            self.log(f"reward_sum/{tag}", reward.sum().item())
            if "step_count" in eval_rollout:
                step_count = eval_rollout["step_count"]
                self.log(f"step_count/{tag}", step_count.max().item())
            del eval_rollout

    def transformed_env(self, base_env: EnvBase) -> EnvBase:
        """Setup transformed environment."""
        return base_env

    def make_env(self) -> EnvBase:
        """You have to implement this method, which has to take no inputs and return
        your environment."""
        raise NotImplementedError("You must implement this method.")

    def _make_env(self) -> EnvBase:
        """Lambda function."""
        env = self.make_env()
        return self.transformed_env(env)

    def state_dict(  # type: ignore
        self,
        *args: ty.Any,
        destination: ty.Optional[ty.Dict[str, ty.Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> ty.Dict[str, ty.Any]:
        """State dict."""
        # Remove env (especially if Serial or Parallel and not plain BaseEnv)
        # Torch is unable to pickle it
        env = self.env
        self.env = None  # type: ignore
        # Now return whatever Torch wanted us to return
        try:
            if destination is not None:
                return super().state_dict(
                    *args,
                    destination=destination,
                    prefix=prefix,
                    keep_vars=keep_vars,
                )
            return super().state_dict(
                *args,
                prefix=prefix,
                keep_vars=keep_vars,
            )
        # Bring `env` back
        finally:
            self.env = env
