# PPOTrainer

*class*torchrl.trainers.algorithms.PPOTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/ppo.html#PPOTrainer)

PPO (Proximal Policy Optimization) trainer implementation.

See also `PPOTrainerConfig` for the
Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This trainer implements the PPO algorithm for training reinforcement learning agents.
It extends [`OnPolicyTrainer`](torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer) with PPO-specific
defaults; see that class for the full list of keyword arguments, covering
advantage estimation (GAE), replay-buffer wiring, collector weight
synchronization and logging.

PPO typically uses multiple epochs of optimization on the same batch of data.
This trainer defaults to 4 epochs, which is a common choice for PPO implementations.

Use [`from_env()`](../trainers_basics.html#torchrl.trainers.algorithms.PPOTrainer.from_env) to construct standard PPO components from an environment,
actor and critic; its docstring includes a complete runnable example.

Note

This trainer requires a configurable environment setup. See the
`configs` module for configuration options.

compute_loss(*sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *method: str | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | tuple[Any, ...]

Evaluate the configured loss through the active execution boundary.

*classmethod*from_env(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*, ***, *actor: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *critic: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *total_frames: int*, *frames_per_batch: int = 1024*, *minibatch_size: int = 256*, *sub_traj_len: int | None = None*, *learning_rate: float = 0.0003*, *value_key: NestedKey = 'state_value'*, *loss_kwargs: Mapping[str, Any] | None = None*, *gae_kwargs: Mapping[str, Any] | None = None*, *collector_kwargs: Mapping[str, Any] | None = None*, ***trainer_kwargs: Any*) → PPOTrainer[[source]](../../_modules/torchrl/trainers/algorithms/ppo.html#PPOTrainer.from_env)

Build a PPO trainer from an environment, actor and critic.

Constructs a [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector),
[`ClipPPOLoss`](torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss), Adam optimizer, GAE and
minibatch sampling. Use the ordinary constructor to supply custom
collectors, losses, optimizers or replay buffers.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - Environment owned by the resulting collector. Training
closes it. The collector installs missing policy primers and
initialization tracking by default.
- **actor** (*TensorDictModuleBase*) - Probabilistic actor returning actions
and their log probabilities.
- **critic** (*TensorDictModuleBase*) - Value network writing `value_key`.
- **total_frames** (*int*) - Total environment transitions to collect. For
closed-loop action deployment these count high-level decisions.
- **frames_per_batch** (*int**,**optional*) - Transitions collected per update,
across all environments. Defaults to 1024.
- **minibatch_size** (*int**,**optional*) - Transitions per optimization step.
Clamped to the collected batch size. Defaults to 256.
- **sub_traj_len** (*int**,**optional*) - Consecutive time steps per recurrent
training window. When set, uses [`BatchSubSampler`](torchrl.trainers.BatchSubSampler.html#torchrl.trainers.BatchSubSampler)
and recurrent-mode GAE instead of flattening time into replay.
The minibatch size must be a multiple of this length.
Defaults to `None` (feedforward PPO).
- **learning_rate** (*float**,**optional*) - Adam learning rate. Defaults to 3e-4.
- **value_key** (*NestedKey**,**optional*) - Critic output key, also configured on
the loss and GAE. Defaults to `"state_value"`.
- **loss_kwargs** (*Mapping**,**optional*) - Extra `ClipPPOLoss` arguments.
Advantage normalization defaults to `True`, or `False` when
GAE already normalizes advantages (e.g. within each task).
- **gae_kwargs** (*Mapping**,**optional*) - Extra GAE arguments. For per-task
normalization, pass `{"group_key": "task_id", "average_gae": True}`.
Recurrent windows default to `shifted=False, deactivate_vmap=True`
because their value networks may depend on recurrent state that
cannot be reconstructed by shifting observations alone.
- **collector_kwargs** (*Mapping**,**optional*) - Extra `Collector` arguments,
such as `policy_device` and `storing_device`.
- ****trainer_kwargs** - Additional [`OnPolicyTrainer`](torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer) options, such
as `num_epochs`, `gamma`, `lmbda`, logging and checkpointing.
Action/reward keys default to the environment's unique keys.
Done/terminated keys use the reward's namespace when available,
otherwise the root namespace; explicit key overrides take precedence.
`frame_skip` defaults to 1 and `clip_norm` to 1.0.

Returns:

Configured trainer; call `train()` to start learning.

Return type:

PPOTrainer

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule, NormalParamExtractor
>>> from torchrl.modules import ProbabilisticActor, TanhNormal
>>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
>>> env = ContinuousActionVecMockEnv()
>>> obs_dim = env.observation_spec["observation"].shape[-1]
>>> action_dim = env.action_spec.shape[-1]
>>> actor = ProbabilisticActor(
... TensorDictModule(
... torch.nn.Sequential(torch.nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor()),
... in_keys=["observation"], out_keys=["loc", "scale"],
... ),
... in_keys=["loc", "scale"], distribution_class=TanhNormal,
... return_log_prob=True,
... )
>>> critic = TensorDictModule(
... torch.nn.Linear(obs_dim, 1), in_keys=["observation"], out_keys=["state_value"],
... )
>>> trainer = PPOTrainer.from_env(
... env, actor=actor, critic=critic, total_frames=32,
... frames_per_batch=16, minibatch_size=8, progress_bar=False,
... )
>>> trainer.train()
```

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function for
legacy torch checkpoints and unified components explicitly saved with
the torch state-dict payload format. Unified checkpoints additionally
accept `strict` to control missing or incompatible components.
Arguments are ignored when `CKPT_BACKEND=memmap`.

Note

Unified state-dict components use TensorDict storage by default and
do not invoke the pickle loader. For explicit torch payloads and
`CKPT_BACKEND=torch` checkpoints, `weights_only=True` is the
default for safer deserialization. Pass `weights_only=False`
explicitly only if the state dict contains custom objects. On
torch < 2.4 the default is `weights_only=False` because the
weights-only unpickler of those versions cannot deserialize the
`torch.device` instances contained in TensorDict state-dicts.

Note

Explicit torch payloads and `CKPT_BACKEND=torch` checkpoints use
`mmap=True` by default. Pass `mmap=False` for legacy pre-zipfile
`torch.save` files or file-like objects. On Windows the default
is `mmap=False` because a mapped checkpoint keeps the file locked,
preventing deletion or re-save.

Note

Unified checkpoint tensors are mapped to CPU by default. Pass an
explicit `map_location` to select another device mapping.

Note

After restoring an independently registered policy component, the
trainer synchronizes the collector once so local policy copies and
remote workers observe the restored learner weights.

Note

`file` may also be a [`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation)
directory, in which case its newest checkpoint is restored.

optim_steps(*batch: ~tensordict.base.TensorDictBase*, ***, *optim_steps_per_batch: int | None | object = <object object>*, *num_epochs: int | object = <object object>*) → None

Run the configured optimization loop for one collected batch.

Keyword overrides are applied only to this call and do not change the
trainer configuration. They are useful for algorithms that need a
one-time optimization schedule while retaining the standard Trainer
hooks and logging behavior.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.

stop_on_signal(*signals: Collection[int] = (Signals.SIGINT, Signals.SIGTERM)*)

Stop training cleanly when the process receives a termination signal.

Wrap `train()` in this context. The first signal calls
`request_stop()`, so the loop finishes the current batch, writes a
final checkpoint when a save destination is configured, shuts the
collector down and returns. A second signal raises
`KeyboardInterrupt`. Previous handlers are restored on exit.

Parameters:

**signals** (*Collection**[**int**]**,**optional*) - signal numbers to handle.
Defaults to `SIGINT` and `SIGTERM`.

Examples

```
>>> with trainer.stop_on_signal(): 
... trainer.train()
```