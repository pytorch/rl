.. currentmodule:: torchrl.trainers

torchrl.trainers package
========================

.. _ref_trainers:

The trainer package provides utilities to write re-usable training scripts. The core idea is to use a
trainer that implements a nested loop, where the outer loop runs the data collection steps and the inner
loop the optimization steps. We believe this fits multiple RL training schemes, such as
on-policy, off-policy, model-based and model-free solutions, offline RL and others.
More particular cases, such as meta-RL algorithms may have training schemes that differ substantially.

The ``trainer.train()`` method can be sketched as follows:

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

   There are 10 hooks that can be used in a trainer loop:

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

   There are 10 hooks that can be used in a trainer loop:

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

There are 10 hooks that can be used in a trainer loop: ``"batch_process"``, ``"pre_optim_steps"``,
``"process_optim_batch"``, ``"post_loss"``, ``"post_steps"``, ``"post_optim"``, ``"pre_steps_log"``,
``"post_steps_log"``, ``"post_optim_log"`` and ``"optimizer"``. They are indicated in the comments where they are applied.
Hooks can be split into 3 categories: **data processing** (``"batch_process"`` and ``"process_optim_batch"``),
**logging** (``"pre_steps_log"``, ``"post_optim_log"`` and ``"post_steps_log"``) and **operations** hook
(``"pre_optim_steps"``, ``"post_loss"``, ``"post_optim"`` and ``"post_steps"``).

- **Data processing** hooks update a tensordict of data. Hooks ``__call__`` method should accept
  a ``TensorDict`` object as input and update it given some strategy.
  Examples of such hooks include Replay Buffer extension (``ReplayBufferTrainer.extend``), data normalization (including normalization
  constants update), data subsampling (:class:``~torchrl.trainers.BatchSubSampler``) and such.

- **Logging** hooks take a batch of data presented as a ``TensorDict`` and write in the logger
  some information retrieved from that data. Examples include the ``LogValidationReward`` hook, the reward
  logger (``LogScalar``) and such. Hooks should return a dictionary (or a None value) containing the
  data to log. The key ``"log_pbar"`` is reserved to boolean values indicating if the logged value
  should be displayed on the progression bar printed on the training log.

- **Operation** hooks are hooks that execute specific operations over the models, data collectors,
  target network updates and such. For instance, syncing the weights of the collectors using ``UpdateWeights``
  or update the priority of the replay buffer using ``ReplayBufferTrainer.update_priority`` are examples
  of operation hooks. They are data-independent (they do not require a ``TensorDict``
  input), they are just supposed to be executed once at every iteration (or every N iterations).

The hooks provided by TorchRL usually inherit from a common abstract class ``TrainerHookBase``,
and all implement three base methods: a ``state_dict`` and ``load_state_dict`` method for
checkpointing and a ``register`` method that registers the hook at the default value in the
trainer. This method takes a trainer and a module name as input. For instance, the following logging
hook is executed every 10 calls to ``"post_optim_log"``:

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
the regular torch backend. This can be controlled via the global variable ``CKPT_BACKEND``:

.. code-block::

    $ CKPT_BACKEND=torchsnapshot python script.py

``CKPT_BACKEND`` defaults to ``torch``. The advantage of torchsnapshot over pytorch
is that it is a more flexible API, which supports distributed checkpointing and
also allows users to load tensors from a file stored on disk to a tensor with a
physical storage (which pytorch currently does not support). This allows, for instance,
to load tensors from and to a replay buffer that would otherwise not fit in memory.

When building a trainer, one can provide a path where the checkpoints are to
be written. With the ``torchsnapshot`` backend, a directory path is expected,
whereas the ``torch`` backend expects a file path (typically a  ``.pt`` file).

.. code-block::

    >>> filepath = "path/to/dir/or/file"
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

The ``Trainer.train()`` method can be used to execute the above loop with all of
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
    LogScalar
    OptimizerHook
    LogValidationReward
    ReplayBufferTrainer
    RewardNormalizer
    SelectKeys
    Trainer
    TrainerHookBase
    UpdateWeights
    TargetNetUpdaterHook
    UTDRHook


Algorithm-specific trainers (Experimental)
------------------------------------------

.. warning::
    The following trainers are experimental/prototype features. The API may change in future versions.
    Please report any issues or feedback to help improve these implementations.

TorchRL provides high-level, algorithm-specific trainers that combine the modular components
into complete training solutions with sensible defaults and comprehensive configuration options.

.. currentmodule:: torchrl.trainers.algorithms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PPOTrainer
    SACTrainer

Algorithm Trainers
~~~~~~~~~~~~~~~~~~

TorchRL provides high-level algorithm trainers that offer complete training solutions with minimal code.
These trainers feature comprehensive configuration systems built on Hydra, enabling both simple usage 
and sophisticated customization.

**Currently Available:**

- :class:`~torchrl.trainers.algorithms.PPOTrainer` - Proximal Policy Optimization
- :class:`~torchrl.trainers.algorithms.SACTrainer` - Soft Actor-Critic

**Key Features:**

- **Complete pipeline**: Environment setup, data collection, and optimization
- **Hydra configuration**: Extensive dataclass-based configuration system
- **Built-in logging**: Rewards, actions, and algorithm-specific metrics
- **Modular design**: Built on existing TorchRL components
- **Minimal code**: Complete SOTA implementations in ~20 lines!

.. warning::
    Algorithm trainers are experimental features. The API may change in future versions. 
    We welcome feedback and contributions to help improve these implementations!

Quick Start Examples
^^^^^^^^^^^^^^^^^^^^

**PPO Training:**

.. code-block:: bash

    # Train PPO on Pendulum-v1 with default settings
    python sota-implementations/ppo_trainer/train.py

**SAC Training:**

.. code-block:: bash

    # Train SAC on a continuous control task
    python sota-implementations/sac_trainer/train.py

**Custom Configuration:**

.. code-block:: bash

    # Override parameters for any algorithm
    python sota-implementations/ppo_trainer/train.py \
        trainer.total_frames=2000000 \
        training_env.create_env_fn.base_env.env_name=HalfCheetah-v4 \
        networks.policy_network.num_cells=[256,256] \
        optimizer.lr=0.0003

**Environment Switching:**

.. code-block:: bash

    # Switch environment and logger for any trainer
    python sota-implementations/sac_trainer/train.py \
        training_env.create_env_fn.base_env.env_name=Walker2d-v4 \
        logger=tensorboard \
        logger.exp_name=sac_walker2d

**View Configuration Options:**

.. code-block:: bash

    # See all available options for any trainer
    python sota-implementations/ppo_trainer/train.py --help
    python sota-implementations/sac_trainer/train.py --help

Universal Configuration System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All algorithm trainers share a unified configuration architecture organized into logical groups:

- **Environment**: ``training_env.create_env_fn.base_env.env_name``, ``training_env.num_workers``
- **Networks**: ``networks.policy_network.num_cells``, ``networks.value_network.num_cells``
- **Training**: ``trainer.total_frames``, ``trainer.clip_norm``, ``optimizer.lr``
- **Data**: ``collector.frames_per_batch``, ``replay_buffer.batch_size``, ``replay_buffer.storage.max_size``
- **Logging**: ``logger.exp_name``, ``logger.project``, ``trainer.log_interval``

**Working Example:**

All trainer implementations follow the same simple pattern:

.. code-block:: python

    import hydra
    from torchrl.trainers.algorithms.configs import *

    @hydra.main(config_path="config", config_name="config", version_base="1.1")
    def main(cfg):
        trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.train()

    if __name__ == "__main__":
        main()

*Complete algorithm training with full configurability in ~20 lines!*

Configuration Classes
^^^^^^^^^^^^^^^^^^^^^

The trainer system uses a hierarchical configuration system with shared components.

.. note::
   The configuration system requires Python 3.10+ due to its use of modern type annotation syntax.

**Algorithm-Specific Trainers:**

- **PPO**: :class:`~torchrl.trainers.algorithms.configs.trainers.PPOTrainerConfig`
- **SAC**: :class:`~torchrl.trainers.algorithms.configs.trainers.SACTrainerConfig`

**Shared Configuration Components:**

- **Environment**: :class:`~torchrl.trainers.algorithms.configs.envs_libs.GymEnvConfig`, :class:`~torchrl.trainers.algorithms.configs.envs.BatchedEnvConfig`
- **Networks**: :class:`~torchrl.trainers.algorithms.configs.modules.MLPConfig`, :class:`~torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig`
- **Data**: :class:`~torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig`, :class:`~torchrl.trainers.algorithms.configs.collectors.MultiaSyncDataCollectorConfig`
- **Objectives**: :class:`~torchrl.trainers.algorithms.configs.objectives.PPOLossConfig`, :class:`~torchrl.trainers.algorithms.configs.objectives.SACLossConfig`
- **Optimizers**: :class:`~torchrl.trainers.algorithms.configs.utils.AdamConfig`, :class:`~torchrl.trainers.algorithms.configs.utils.AdamWConfig`
- **Logging**: :class:`~torchrl.trainers.algorithms.configs.logging.WandbLoggerConfig`, :class:`~torchrl.trainers.algorithms.configs.logging.TensorboardLoggerConfig`

Algorithm-Specific Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PPOTrainer:**

- On-policy learning with advantage estimation
- Policy clipping and value function optimization
- Configurable number of epochs per batch
- Built-in GAE (Generalized Advantage Estimation)

**SACTrainer:**

- Off-policy learning with replay buffer
- Entropy-regularized policy optimization
- Target network soft updates
- Continuous action space optimization

**Future Development:**

The trainer system is actively expanding. Upcoming features include:

- Additional algorithms: TD3, DQN, A2C, DDPG, and more
- Enhanced distributed training support
- Advanced configuration validation and error reporting
- Integration with more TorchRL ecosystem components

See the complete `configuration system documentation <https://github.com/pytorch/rl/tree/main/torchrl/trainers/algorithms/configs>`_ for all available options and examples.


Builders
--------

.. currentmodule:: torchrl.trainers.helpers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_collector_offpolicy
    make_collector_onpolicy
    make_dqn_loss
    make_replay_buffer
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

.. _ref_loggers:

.. currentmodule:: torchrl.record.loggers

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    Logger
    csv.CSVLogger
    mlflow.MLFlowLogger
    tensorboard.TensorboardLogger
    wandb.WandbLogger
    get_logger
    generate_exp_name


Recording utils
---------------

Recording utils are detailed :ref:`here <Environment-Recorders>`.

.. currentmodule:: torchrl.record

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    VideoRecorder
    TensorDictRecorder
    PixelRenderTransform
