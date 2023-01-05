.. currentmodule:: torchrl.trainers

torchrl.trainers package
========================

The trainer package provides utilities to write re-usable training scripts. The core idea is to use a
trainer that implements a nested loop, where the outer loop runs the data collection steps and the inner
loop the optimization steps. We believe this fits multiple RL training schemes, such as
on-policy, off-policy, model-based and model-free solutions, offline RL and others.
More particular cases, such as meta-RL algorithms may have training schemes that differ substentially.

The :obj:`trainer.train()` method can be sketched as follows:

.. code-block::
   :caption: Trainer loops

        >>> for batch in collector:
        ...     batch = self._process_batch_hook(batch)  # "batch_process"
        ...     self._pre_steps_log_hook(batch)  # "pre_steps_log"
        ...     self._pre_optim_hook()  # "pre_optim_steps"
        ...     for j in range(self.optim_steps_per_batch):
        ...         sub_batch = self._process_optim_batch_hook(batch)  # "process_optim_batch"
        ...         losses = self.loss_module(sub_batch)
        ...         self._post_loss_hook(sub_batch)  # "post_loss"
        ...         self.optimizer.step()
        ...         self.optimizer.zero_grad()
        ...         self._post_optim_hook()  # "post_optim"
        ...         self._post_optim_log(sub_batch)  # "post_optim_log"
        ...     self._post_steps_hook()  # "post_steps"
        ...     self._post_steps_log_hook(batch)  #  "post_steps_log"

There are 10 hooks that can be used in a trainer loop: :obj:`"batch_process"`, :obj:`"pre_optim_steps"`,
:obj:`"process_optim_batch"`, :obj:`"post_loss"`, :obj:`"post_steps"`, :obj:`"post_optim"`, :obj:`"pre_steps_log"`,
:obj:`"post_steps_log"`, :obj:`"post_optim_log"` and :obj:`"optimizer"`. They are indicated in the comments where they are applied.
Hooks can be split into 3 categories: **data processing** (:obj:`"batch_process"` and :obj:`"process_optim_batch"`),
**logging** (:obj:`"pre_steps_log"`, :obj:`"post_optim_log"` and :obj:`"post_steps_log"`) and **operations** hook
(:obj:`"pre_optim_steps"`, :obj:`"post_loss"`, :obj:`"post_optim"` and :obj:`"post_steps"`).

- **Data processing** hooks update a tensordict of data. Hooks :obj:`__call__` method should accept
  a :obj:`TensorDict` object as input and update it given some strategy.
  Examples of such hooks include Replay Buffer extension (:obj:`ReplayBufferTrainer.extend`), data normalization (including normalization
  constants update), data subsampling (:doc:`BatchSubSampler`) and such.

- **Logging** hooks take a batch of data presented as a :obj:`TensorDict` and write in the logger
  some information retrieved from that data. Examples include the :obj:`Recorder` hook, the reward
  logger (:obj:`LogReward`) and such. Hooks should return a dictionary (or a None value) containing the
  data to log. The key :obj:`"log_pbar"` is reserved to boolean values indicating if the logged value
  should be displayed on the progression bar printed on the training log.

- **Operation** hooks are hooks that execute specific operations over the models, data collectors,
  target network updates and such. For instance, syncing the weights of the collectors using :obj:`UpdateWeights`
  or update the priority of the replay buffer using :obj:`ReplayBufferTrainer.update_priority` are examples
  of operation hooks. They are data-independent (they do not require a :obj:`TensorDict`
  input), they are just supposed to be executed once at every iteration (or every N iterations).

The hooks provided by TorchRL usually inherit from a common abstract class :obj:`TrainerHookBase`,
and all implement three base methods: a :obj:`state_dict` and :obj:`load_state_dict` method for
checkpointing and a :obj:`register` method that registers the hook at the default value in the
trainer. This method takes a trainer and a module name as input. For instance, the following logging
hook is executed every 10 calls to :obj:`"post_optim_log"`:

.. code-block::

        >>> class LoggingHook(TrainerHookBase):
        ...     def __init__(self):
        ...         self.counter = 0
        ...
        ...     def register(self, trainer, name):
        ...         trainer.register_module(self, "logging_hook")
        ...         trainer.register_op("post_optim_log", self)
        ...
        ...     def save_dict(self):
        ...         return {"counter": self.counter}
        ...
        ...     def load_state_dict(self, state_dict):
        ...         self.counter = state_dict["counter"]
        ...
        ...     def __call__(self, batch):
        ...         if self.counter % 10 == 0:
        ...             self.counter += 1
        ...             out = {"some_value": batch["some_value"].item(), "log_pbar": False}
        ...         else:
        ...             out = None
        ...         self.counter += 1
        ...         return out

Checkpointing
-------------

The trainer class and hooks support checkpointing, which can be achieved either
using the `torchsnapshot <https://github.com/pytorch/torchsnapshot/>`_ backend or
the regular torch backend. This can be controlled via the global variable :obj:`CKPT_BACKEND`:

.. code-block::

    $ CKPT_BACKEND=torch python script.py

which defaults to :obj:`torchsnapshot`. The advantage of torchsnapshot over pytorch
is that it is a more flexible API, which supports distributed checkpointing and
also allows users to load tensors from a file stored on disk to a tensor with a
physical storage (which pytorch currently does not support). This allows, for instance,
to load tensors from and to a replay buffer that would otherwise not fit in memory.

When building a trainer, one can provide a file path where the checkpoints are to
be written. With the :obj:`torchsnapshot` backend, a directory path is expected,
whereas the :obj:`torch` backend expects a file path (typically a  :obj:`.pt` file).

.. code-block::

    >>> filepath = "path/to/dir/"
    >>> trainer = Trainer(
    ...     collector=collector,
    ...     total_frames=total_frames,
    ...     frame_skip=frame_skip,
    ...     loss_module=loss_module,
    ...     optimizer=optimizer,
    ...     save_trainer_file=filepath,
    ... )
    >>> select_keys = SelectKeys(["action", "observation"])
    >>> select_keys.register(trainer)
    >>> # to save to a path
    >>> trainer.save_trainer(True)
    >>> # to load from a path
    >>> trainer.load_from_file(filepath)

The :obj:`Trainer.train()` method can be used to execute the above loop with all of
its hooks, although using the :obj:`Trainer` class for its checkpointing capability
only is also a perfectly valid use.


Trainer and hooks
-----------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    BatchSubSampler
    ClearCudaCache
    CountFramesLog
    LogReward
    OptimizerHook
    Recorder
    ReplayBuffer
    RewardNormalizer
    SelectKeys
    Trainer
    TrainerHookBase
    UpdateWeights


Builders
--------

.. currentmodule:: torchrl.trainers.helpers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_a2c_loss
    make_a2c_model
    make_collector_offpolicy
    make_collector_onpolicy
    make_ddpg_actor
    make_ddpg_loss
    make_dqn_actor
    make_dqn_loss
    make_ppo_loss
    make_ppo_model
    make_redq_loss
    make_redq_model
    make_replay_buffer
    make_sac_loss
    make_sac_model
    make_target_updater
    make_trainer
    parallel_env_constructor
    sync_async_collector
    sync_sync_collector
    transformed_env_constructor

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    correct_for_frame_skip
    get_stats_random_rollout

Loggers
-------

.. currentmodule:: torchrl.trainers.loggers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    Logger
    CSVLogger
    MLFlowLogger
    TensorboardLogger
    WandbLogger
    get_logger
    generate_exp_name
