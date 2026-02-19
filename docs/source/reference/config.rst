.. currentmodule:: torchrl.trainers.algorithms.configs

TorchRL Configuration System
============================

TorchRL provides a powerful configuration system built on top of `Hydra <https://hydra.cc/>`_ that enables you to easily configure 
and run reinforcement learning experiments. This system uses structured dataclass-based configurations that can be composed, overridden, and extended.

The advantages of using a configuration system are:
- Quick and easy to get started: provide your task and let the system handle the rest
- Get a glimpse of the available options and their default values in one go: ``python sota-implementations/ppo_trainer/train.py --help`` will show you all the available options and their default values
- Easy to override and extend: you can override any option in the configuration file, and you can also extend the configuration file with your own custom configurations
- Easy to share and reproduce: you can share your configuration file with others, and they can reproduce your results by simply running the same command.
- Easy to version control: you can easily version control your configuration file

Quick Start with a Simple Example
---------------------------------

Let's start with a simple example that creates a Gym environment. Here's a minimal configuration file:

.. code-block:: yaml

    # config.yaml
    defaults:
      - env@training_env: gym

    training_env:
      env_name: CartPole-v1

This configuration has two main parts:

**1. The** ``defaults`` **section**

The ``defaults`` section tells Hydra which configuration groups to include. In this case:

- ``env@training_env: gym`` means "use the 'gym' configuration from the 'env' group for the 'training_env' target"

This is equivalent to including a predefined configuration for Gym environments, which sets up the proper target class and default parameters.

**2. The configuration override**

The ``training_env`` section allows you to override or specify parameters for the selected configuration:

- ``env_name: CartPole-v1`` sets the specific environment name

Configuration Categories and Groups
-----------------------------------

TorchRL organizes configurations into several categories using the ``@`` syntax for targeted configuration:

- ``env@<target>``: Environment configurations (Gym, DMControl, Brax, etc.) as well as batched environments
- ``transform@<target>``: Transform configurations (observation/reward processing)
- ``model@<target>``: Model configurations (policy and value networks)
- ``network@<target>``: Neural network configurations (MLP, ConvNet)
- ``collector@<target>``: Data collection configurations
- ``replay_buffer@<target>``: Replay buffer configurations
- ``storage@<target>``: Storage backend configurations
- ``sampler@<target>``: Sampling strategy configurations
- ``writer@<target>``: Writer strategy configurations
- ``trainer@<target>``: Training loop configurations
- ``optimizer@<target>``: Optimizer configurations
- ``loss@<target>``: Loss function configurations
- ``logger@<target>``: Logging configurations

The ``@<target>`` syntax allows you to assign configurations to specific locations in your config structure.

More Complex Example: Parallel Environment with Transforms
----------------------------------------------------------

Here's a more complex example that creates a parallel environment with multiple transforms applied to each worker:

.. code-block:: yaml

    defaults:
      - env@training_env: batched_env
      - env@training_env.create_env_fn: transformed_env
      - env@training_env.create_env_fn.base_env: gym
      - transform@training_env.create_env_fn.transform: compose
      - transform@transform0: noop_reset
      - transform@transform1: step_counter

    # Transform configurations
    transform0:
      noops: 30
      random: true

    transform1:
      max_steps: 200
      step_count_key: "step_count"

    # Environment configuration
    training_env:
      num_workers: 4
      create_env_fn:
        base_env:
          env_name: Pendulum-v1
        transform:
          transforms:
            - ${transform0}
            - ${transform1}
        _partial_: true

**What this configuration creates:**

This configuration builds a **parallel environment with 4 workers**, where each worker runs a **Pendulum-v1 environment with two transforms applied**:

1. **Parallel Environment Structure**: 
   - ``batched_env`` creates a parallel environment that runs multiple environment instances
   - ``num_workers: 4`` means 4 parallel environment processes

2. **Individual Environment Construction** (repeated for each of the 4 workers):
   - **Base Environment**: ``gym`` with ``env_name: Pendulum-v1`` creates a Pendulum environment
   - **Transform Layer 1**: ``noop_reset`` performs 30 random no-op actions at episode start
   - **Transform Layer 2**: ``step_counter`` limits episodes to 200 steps and tracks step count
   - **Transform Composition**: ``compose`` combines both transforms into a single transformation

3. **Final Result**: 4 parallel Pendulum environments, each with:
   - Random no-op resets (0-30 actions at start)
   - Maximum episode length of 200 steps
   - Step counting functionality

**Key Configuration Concepts:**

1. **Nested targeting**: ``env@training_env.create_env_fn.base_env: gym`` places a gym config deep inside the structure
2. **Function factories**: ``_partial_: true`` creates a function that can be called multiple times (once per worker)
3. **Transform composition**: Multiple transforms are combined and applied to each environment instance
4. **Variable interpolation**: ``${transform0}`` and ``${transform1}`` reference the separately defined transform configurations

Getting Available Options
-------------------------

To explore all available configurations and their parameters, one can use the ``--help`` flag with any TorchRL script:

.. code-block:: bash

    python sota-implementations/ppo_trainer/train.py --help

This shows all configuration groups and their options, making it easy to discover what's available. It should print something like this:

.. code-block:: bash


Complete Training Example
-------------------------

Here's a complete configuration for PPO training:

.. code-block:: yaml

    defaults:
      - env@training_env: batched_env
      - env@training_env.create_env_fn: gym
      - model@models.policy_model: tanh_normal
      - model@models.value_model: value
      - network@networks.policy_network: mlp
      - network@networks.value_network: mlp
      - collector: sync
      - replay_buffer: base
      - storage: tensor
      - sampler: without_replacement
      - writer: round_robin
      - trainer: ppo
      - optimizer: adam
      - loss: ppo
      - logger: wandb

    # Network configurations
    networks:
      policy_network:
        out_features: 2
        in_features: 4
        num_cells: [128, 128]

      value_network:
        out_features: 1
        in_features: 4
        num_calls: [128, 128]

    # Model configurations
    models:
      policy_model:
        network: ${networks.policy_network}
        in_keys: ["observation"]
        out_keys: ["action"]

      value_model:
        network: ${networks.value_network}
        in_keys: ["observation"]
        out_keys: ["state_value"]

    # Environment
    training_env:
      num_workers: 2
      create_env_fn:
        env_name: CartPole-v1
        _partial_: true

    # Training components
    trainer:
      collector: ${collector}
      optimizer: ${optimizer}
      loss_module: ${loss}
      logger: ${logger}
      total_frames: 100000

    collector:
      create_env_fn: ${training_env}
      policy: ${models.policy_model}
      frames_per_batch: 1024

    optimizer:
      lr: 0.001

    loss:
      actor_network: ${models.policy_model}
      critic_network: ${models.value_model}

    logger:
      exp_name: my_experiment

Running Experiments
-------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

    # Use default configuration
    python sota-implementations/ppo_trainer/train.py

    # Override specific parameters
    python sota-implementations/ppo_trainer/train.py optimizer.lr=0.0001

    # Change environment
    python sota-implementations/ppo_trainer/train.py training_env.create_env_fn.env_name=Pendulum-v1

    # Use different collector
    python sota-implementations/ppo_trainer/train.py collector=async

Hyperparameter Sweeps
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Sweep over learning rates
    python sota-implementations/ppo_trainer/train.py --multirun optimizer.lr=0.0001,0.001,0.01

    # Multiple parameter sweep
    python sota-implementations/ppo_trainer/train.py --multirun \
      optimizer.lr=0.0001,0.001 \
      training_env.num_workers=2,4,8

Custom Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use custom config file
    python sota-implementations/ppo_trainer/train.py --config-name my_custom_config

Configuration Store Implementation Details
------------------------------------------

Under the hood, TorchRL uses Hydra's ConfigStore to register all configuration classes. This provides type safety, validation, and IDE support. The registration happens automatically when you import the configs module:

.. code-block:: python

    from hydra.core.config_store import ConfigStore
    from torchrl.trainers.algorithms.configs import *

    cs = ConfigStore.instance()

    # Environments
    cs.store(group="env", name="gym", node=GymEnvConfig)
    cs.store(group="env", name="batched_env", node=BatchedEnvConfig)

    # Models  
    cs.store(group="model", name="tanh_normal", node=TanhNormalModelConfig)
    # ... and many more

Available Configuration Classes
-------------------------------

Base Classes
~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.common

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    ConfigBase

Environment Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.envs

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    EnvConfig
    BatchedEnvConfig
    TransformedEnvConfig

Environment Library Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.envs_libs

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    EnvLibsConfig
    GymEnvConfig
    DMControlEnvConfig
    BraxEnvConfig
    HabitatEnvConfig
    IsaacGymEnvConfig
    JumanjiEnvConfig
    MeltingpotEnvConfig
    MOGymEnvConfig
    MultiThreadedEnvConfig
    OpenEnvEnvConfig
    OpenMLEnvConfig
    OpenSpielEnvConfig
    PettingZooEnvConfig
    RoboHiveEnvConfig
    SMACv2EnvConfig
    UnityMLAgentsEnvConfig
    VmasEnvConfig

Model and Network Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    ModelConfig
    NetworkConfig
    MLPConfig
    ConvNetConfig
    TensorDictModuleConfig
    TanhNormalModelConfig
    ValueModelConfig

Transform Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.transforms

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    TransformConfig
    ComposeConfig
    NoopResetEnvConfig
    StepCounterConfig
    DoubleToFloatConfig
    ToTensorImageConfig
    ClipTransformConfig
    ResizeConfig
    CenterCropConfig
    CropConfig
    FlattenObservationConfig
    GrayScaleConfig
    ObservationNormConfig
    CatFramesConfig
    RewardClippingConfig
    RewardScalingConfig
    BinarizeRewardConfig
    TargetReturnConfig
    VecNormConfig
    FrameSkipTransformConfig
    DeviceCastTransformConfig
    DTypeCastTransformConfig
    UnsqueezeTransformConfig
    SqueezeTransformConfig
    PermuteTransformConfig
    CatTensorsConfig
    StackConfig
    DiscreteActionProjectionConfig
    TensorDictPrimerConfig
    PinMemoryTransformConfig
    RewardSumConfig
    ExcludeTransformConfig
    SelectTransformConfig
    TimeMaxPoolConfig
    RandomCropTensorDictConfig
    InitTrackerConfig
    RenameTransformConfig
    Reward2GoTransformConfig
    ActionMaskConfig
    VecGymEnvTransformConfig
    BurnInTransformConfig
    SignTransformConfig
    RemoveEmptySpecsConfig
    BatchSizeTransformConfig
    AutoResetTransformConfig
    ActionDiscretizerConfig
    TrajCounterConfig
    LineariseRewardsConfig
    ConditionalSkipConfig
    MultiActionConfig
    TimerConfig
    ConditionalPolicySwitchConfig
    FiniteTensorDictCheckConfig
    UnaryTransformConfig
    HashConfig
    TokenizerConfig
    EndOfLifeTransformConfig
    MultiStepTransformConfig
    KLRewardTransformConfig
    R3MTransformConfig
    VC1TransformConfig
    VIPTransformConfig
    VIPRewardTransformConfig
    VecNormV2Config

Data Collection Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.collectors

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    CollectorConfig
    AsyncCollectorConfig
    MultiSyncCollectorConfig
    MultiAsyncCollectorConfig

Replay Buffer and Storage Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.data

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    ReplayBufferConfig
    TensorDictReplayBufferConfig
    RandomSamplerConfig
    SamplerWithoutReplacementConfig
    PrioritizedSamplerConfig
    SliceSamplerConfig
    SliceSamplerWithoutReplacementConfig
    ListStorageConfig
    TensorStorageConfig
    LazyTensorStorageConfig
    LazyMemmapStorageConfig
    LazyStackStorageConfig
    StorageEnsembleConfig
    RoundRobinWriterConfig
    StorageEnsembleWriterConfig

Training and Optimization Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.trainers

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    TrainerConfig
    PPOTrainerConfig

.. currentmodule:: torchrl.trainers.algorithms.configs.objectives

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    LossConfig
    PPOLossConfig

.. currentmodule:: torchrl.trainers.algorithms.configs.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    AdamConfig
    AdamWConfig
    AdamaxConfig
    AdadeltaConfig
    AdagradConfig
    ASGDConfig
    LBFGSConfig
    LionConfig
    NAdamConfig
    RAdamConfig
    RMSpropConfig
    RpropConfig
    SGDConfig
    SparseAdamConfig

Logging Configurations
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.trainers.algorithms.configs.logging

.. autosummary::
    :toctree: generated/
    :template: rl_template_class.rst

    LoggerConfig
    WandbLoggerConfig
    TensorboardLoggerConfig
    CSVLoggerConfig

Creating Custom Configurations
------------------------------

You can create custom configuration classes by inheriting from the appropriate base classes:

.. code-block:: python

    from dataclasses import dataclass
    from torchrl.trainers.algorithms.configs.envs_libs import EnvLibsConfig

    @dataclass
    class MyCustomEnvConfig(EnvLibsConfig):
        _target_: str = "my_module.MyCustomEnv"
        env_name: str = "MyEnv-v1"
        custom_param: float = 1.0
        
        def __post_init__(self):
            super().__post_init__()

    # Register with ConfigStore
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(group="env", name="my_custom", node=MyCustomEnvConfig)

Best Practices
--------------

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Use Defaults**: Leverage the ``defaults`` section to compose configurations
3. **Override Sparingly**: Only override what you need to change
4. **Validate Configurations**: Test that your configurations instantiate correctly
5. **Version Control**: Keep your configuration files under version control
6. **Use Variable Interpolation**: Use ``${variable}`` syntax to avoid duplication

Future Extensions
-----------------

As TorchRL adds more algorithms beyond PPO (such as SAC, TD3, DQN), the configuration system will expand with:

- New trainer configurations (e.g., ``SACTrainerConfig``, ``TD3TrainerConfig``)
- Algorithm-specific loss configurations
- Specialized collector configurations for different algorithms
- Additional environment and model configurations

The modular design ensures easy integration while maintaining backward compatibility.
