# TorchRL Configuration System

TorchRL provides a powerful configuration system built on top of [Hydra](https://hydra.cc/) that enables you to easily configure
and run reinforcement learning experiments. This system uses structured dataclass-based configurations that can be composed, overridden, and extended.

The advantages of using a configuration system are:
- Quick and easy to get started: provide your task and let the system handle the rest
- Get a glimpse of the available options and their default values in one go: `python sota-implementations/ppo_trainer/train.py --help` will show you all the available options and their default values
- Easy to override and extend: you can override any option in the configuration file, and you can also extend the configuration file with your own custom configurations
- Easy to share and reproduce: you can share your configuration file with others, and they can reproduce your results by simply running the same command.
- Easy to version control: you can easily version control your configuration file

## Quick Start with a Simple Example

Let's start with a simple example that creates a Gym environment. Here's a minimal configuration file:

```
# config.yaml
defaults:
 - env@training_env: gym

training_env:
 env_name: CartPole-v1
```

This configuration has two main parts:

**1. The** `defaults` **section**

The `defaults` section tells Hydra which configuration groups to include. In this case:

- `env@training_env: gym` means "use the 'gym' configuration from the 'env' group for the 'training_env' target"

This is equivalent to including a predefined configuration for Gym environments, which sets up the proper target class and default parameters.

**2. The configuration override**

The `training_env` section allows you to override or specify parameters for the selected configuration:

- `env_name: CartPole-v1` sets the specific environment name

## Configuration Categories and Groups

TorchRL organizes configurations into several categories using the `@` syntax for targeted configuration:

- `env@<target>`: Environment configurations (Gym, DMControl, Brax, etc.) as well as batched environments
- `transform@<target>`: Transform configurations (observation/reward processing)
- `model@<target>`: Model configurations (policy and value networks)
- `network@<target>`: Neural network configurations (MLP, ConvNet)
- `collector@<target>`: Data collection configurations
- `replay_buffer@<target>`: Replay buffer configurations
- `storage@<target>`: Storage backend configurations
- `sampler@<target>`: Sampling strategy configurations
- `writer@<target>`: Writer strategy configurations
- `trainer@<target>`: Training loop configurations
- `hook@<target>`: Trainer hook configurations
- `optimizer@<target>`: Optimizer configurations
- `loss@<target>`: Loss function configurations
- `logger@<target>`: Logging configurations

The `@<target>` syntax allows you to assign configurations to specific locations in your config structure.

## More Complex Example: Parallel Environment with Transforms

Here's a more complex example that creates a parallel environment with multiple transforms applied to each worker:

```
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
```

**What this configuration creates:**

This configuration builds a **parallel environment with 4 workers**, where each worker runs a **Pendulum-v1 environment with two transforms applied**:

1. **Parallel Environment Structure**:
- `batched_env` creates a parallel environment that runs multiple environment instances
- `num_workers: 4` means 4 parallel environment processes
2. **Individual Environment Construction** (repeated for each of the 4 workers):
- **Base Environment**: `gym` with `env_name: Pendulum-v1` creates a Pendulum environment
- **Transform Layer 1**: `noop_reset` performs 30 random no-op actions at episode start
- **Transform Layer 2**: `step_counter` limits episodes to 200 steps and tracks step count
- **Transform Composition**: `compose` combines both transforms into a single transformation
3. **Final Result**: 4 parallel Pendulum environments, each with:
- Random no-op resets (0-30 actions at start)
- Maximum episode length of 200 steps
- Step counting functionality

**Key Configuration Concepts:**

1. **Nested targeting**: `env@training_env.create_env_fn.base_env: gym` places a gym config deep inside the structure
2. **Function factories**: `_partial_: true` creates a function that can be called multiple times (once per worker)
3. **Transform composition**: Multiple transforms are combined and applied to each environment instance
4. **Variable interpolation**: `${transform0}` and `${transform1}` reference the separately defined transform configurations

## Getting Available Options

To explore all available configurations and their parameters, one can use the `--help` flag with any TorchRL script:

```
python sota-implementations/ppo_trainer/train.py --help
```

This shows all configuration groups and their options, making it easy to discover what's available. It should print something like this:

```

```

## Complete Training Example

Here's a complete configuration for PPO training:

```
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
```

## Running Experiments

### Basic Usage

```
# Use default configuration
python sota-implementations/ppo_trainer/train.py

# Override specific parameters
python sota-implementations/ppo_trainer/train.py optimizer.lr=0.0001

# Change environment
python sota-implementations/ppo_trainer/train.py training_env.create_env_fn.env_name=Pendulum-v1

# Use different collector
python sota-implementations/ppo_trainer/train.py collector=async
```

### Hyperparameter Sweeps

```
# Sweep over learning rates
python sota-implementations/ppo_trainer/train.py --multirun optimizer.lr=0.0001,0.001,0.01

# Multiple parameter sweep
python sota-implementations/ppo_trainer/train.py --multirun \
 optimizer.lr=0.0001,0.001 \
 training_env.num_workers=2,4,8
```

### Custom Configuration Files

```
# Use custom config file
python sota-implementations/ppo_trainer/train.py --config-name my_custom_config
```

## Configuration Store Implementation Details

Under the hood, TorchRL uses Hydra's ConfigStore to register all configuration classes. This provides type safety, validation, and IDE support. The registration happens automatically when you import the configs module:

```
from hydra.core.config_store import ConfigStore
from torchrl.trainers.algorithms.configs import *

cs = ConfigStore.instance()

# Environments
cs.store(group="env", name="gym", node=GymEnvConfig)
cs.store(group="env", name="batched_env", node=BatchedEnvConfig)

# Models
cs.store(group="model", name="tanh_normal", node=TanhNormalModelConfig)
# ... and many more
```

## Available Configuration Classes

### Base Classes

| [`ConfigBase`](generated/torchrl.trainers.algorithms.configs.common.ConfigBase.html#torchrl.trainers.algorithms.configs.common.ConfigBase)() | Abstract base class for all configuration classes. |
| --- | --- |

### Environment Configurations

| [`EnvConfig`](generated/torchrl.trainers.algorithms.configs.envs.EnvConfig.html#torchrl.trainers.algorithms.configs.envs.EnvConfig)([_partial_]) | Base configuration class for environments. |
| --- | --- |
| [`BatchedEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs.BatchedEnvConfig.html#torchrl.trainers.algorithms.configs.envs.BatchedEnvConfig)(_partial_, create_env_fn, ...) | Configuration for batched environments. |
| [`TransformedEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs.TransformedEnvConfig.html#torchrl.trainers.algorithms.configs.envs.TransformedEnvConfig)([_partial_, base_env, ...]) | Configuration for transformed environments. |

### Environment Library Configurations

| [`EnvLibsConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.EnvLibsConfig.html#torchrl.trainers.algorithms.configs.envs_libs.EnvLibsConfig)([_partial_]) | Base configuration class for environment libs. |
| --- | --- |
| [`GymEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.GymEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.GymEnvConfig)([_partial_, env_name, ...]) | Configuration for GymEnv environment. |
| [`DMControlEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.DMControlEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.DMControlEnvConfig)([_partial_, env_name, ...]) | Configuration for DMControlEnv environment. |
| [`BraxEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.BraxEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.BraxEnvConfig)([_partial_, env_name, ...]) | Configuration for BraxEnv environment. |
| [`HabitatEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.HabitatEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.HabitatEnvConfig)([_partial_, env_name, ...]) | Configuration for HabitatEnv environment. |
| [`IsaacGymEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.IsaacGymEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.IsaacGymEnvConfig)([_partial_, env_name, ...]) | Configuration for IsaacGymEnv environment. |
| [`JumanjiEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.JumanjiEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.JumanjiEnvConfig)([_partial_, env_name, ...]) | Configuration for JumanjiEnv environment. |
| [`MeltingpotEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.MeltingpotEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.MeltingpotEnvConfig)([_partial_, env_name, ...]) | Configuration for MeltingpotEnv environment. |
| [`MOGymEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.MOGymEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.MOGymEnvConfig)([_partial_, env_name, ...]) | Configuration for MOGymEnv environment. |
| [`MultiThreadedEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.MultiThreadedEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.MultiThreadedEnvConfig)([_partial_, ...]) | Configuration for MultiThreadedEnv environment. |
| [`OpenMLEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.OpenMLEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.OpenMLEnvConfig)([_partial_, env_name, ...]) | Configuration for OpenMLEnv environment. |
| [`OpenSpielEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.OpenSpielEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.OpenSpielEnvConfig)([_partial_, env_name, ...]) | Configuration for OpenSpielEnv environment. |
| [`PettingZooEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.PettingZooEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.PettingZooEnvConfig)([_partial_, env_name, ...]) | Configuration for PettingZooEnv environment. |
| [`RoboHiveEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.RoboHiveEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.RoboHiveEnvConfig)([_partial_, env_name, ...]) | Configuration for RoboHiveEnv environment. |
| [`SMACv2EnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.SMACv2EnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.SMACv2EnvConfig)([_partial_, env_name, ...]) | Configuration for SMACv2Env environment. |
| [`UnityMLAgentsEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.UnityMLAgentsEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.UnityMLAgentsEnvConfig)([_partial_, ...]) | Configuration for UnityMLAgentsEnv environment. |
| [`VmasEnvConfig`](generated/torchrl.trainers.algorithms.configs.envs_libs.VmasEnvConfig.html#torchrl.trainers.algorithms.configs.envs_libs.VmasEnvConfig)(_partial_, scenario, num_envs, ...) | Configuration for VmasEnv environment. |

### Model and Network Configurations

| [`ModelConfig`](generated/torchrl.trainers.algorithms.configs.modules.ModelConfig.html#torchrl.trainers.algorithms.configs.modules.ModelConfig)([_partial_, in_keys, out_keys, ...]) | Parent class to configure a model. |
| --- | --- |
| [`NetworkConfig`](generated/torchrl.trainers.algorithms.configs.modules.NetworkConfig.html#torchrl.trainers.algorithms.configs.modules.NetworkConfig)([_partial_]) | Parent class to configure a network. |
| [`MLPConfig`](generated/torchrl.trainers.algorithms.configs.modules.MLPConfig.html#torchrl.trainers.algorithms.configs.modules.MLPConfig)(_partial_, in_features, ...) | A class to configure a multi-layer perceptron. |
| [`ConvNetConfig`](generated/torchrl.trainers.algorithms.configs.modules.ConvNetConfig.html#torchrl.trainers.algorithms.configs.modules.ConvNetConfig)(_partial_, in_features, depth, ...) | A class to configure a convolutional network. |
| [`TensorDictModuleConfig`](generated/torchrl.trainers.algorithms.configs.modules.TensorDictModuleConfig.html#torchrl.trainers.algorithms.configs.modules.TensorDictModuleConfig)([_partial_, in_keys, ...]) | A class to configure a TensorDictModule. |
| [`TanhNormalModelConfig`](generated/torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig.html#torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig)([_partial_, in_keys, ...]) | A class to configure a TanhNormal model. |
| [`ValueModelConfig`](generated/torchrl.trainers.algorithms.configs.modules.ValueModelConfig.html#torchrl.trainers.algorithms.configs.modules.ValueModelConfig)([_partial_, in_keys, ...]) | A class to configure a Value model. |
| [`QValueModelConfig`](generated/torchrl.trainers.algorithms.configs.modules.QValueModelConfig.html#torchrl.trainers.algorithms.configs.modules.QValueModelConfig)([_partial_, in_keys, ...]) | A class to configure a QValueActor model. |
| [`TanhModuleConfig`](generated/torchrl.trainers.algorithms.configs.modules.TanhModuleConfig.html#torchrl.trainers.algorithms.configs.modules.TanhModuleConfig)([_partial_, in_keys, ...]) | A class to configure a TanhModule. |
| [`TensorDictSequentialConfig`](generated/torchrl.trainers.algorithms.configs.modules.TensorDictSequentialConfig.html#torchrl.trainers.algorithms.configs.modules.TensorDictSequentialConfig)([_partial_, ...]) | A class to configure a TensorDictSequential. |
| [`AdditiveGaussianModuleConfig`](generated/torchrl.trainers.algorithms.configs.modules.AdditiveGaussianModuleConfig.html#torchrl.trainers.algorithms.configs.modules.AdditiveGaussianModuleConfig)([_partial_, ...]) | A class to configure an AdditiveGaussianModule. |

### Transform Configurations

| [`TransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TransformConfig.html#torchrl.trainers.algorithms.configs.transforms.TransformConfig)() | Base configuration class for transforms. |
| --- | --- |
| [`ComposeConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ComposeConfig.html#torchrl.trainers.algorithms.configs.transforms.ComposeConfig)([transforms, _target_]) | Configuration for Compose transform. |
| [`NoopResetEnvConfig`](generated/torchrl.trainers.algorithms.configs.transforms.NoopResetEnvConfig.html#torchrl.trainers.algorithms.configs.transforms.NoopResetEnvConfig)([noops, random, _target_]) | Configuration for NoopResetEnv transform. |
| [`StepCounterConfig`](generated/torchrl.trainers.algorithms.configs.transforms.StepCounterConfig.html#torchrl.trainers.algorithms.configs.transforms.StepCounterConfig)([max_steps, ...]) | Configuration for StepCounter transform. |
| [`DoubleToFloatConfig`](generated/torchrl.trainers.algorithms.configs.transforms.DoubleToFloatConfig.html#torchrl.trainers.algorithms.configs.transforms.DoubleToFloatConfig)([in_keys, out_keys, ...]) | Configuration for DoubleToFloat transform. |
| [`ToTensorImageConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ToTensorImageConfig.html#torchrl.trainers.algorithms.configs.transforms.ToTensorImageConfig)([from_int, unsqueeze, ...]) | Configuration for ToTensorImage transform. |
| [`ClipTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ClipTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.ClipTransformConfig)([in_keys, out_keys, ...]) | Configuration for ClipTransform. |
| [`ResizeConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ResizeConfig.html#torchrl.trainers.algorithms.configs.transforms.ResizeConfig)([w, h, interpolation, in_keys, ...]) | Configuration for Resize transform. |
| [`CenterCropConfig`](generated/torchrl.trainers.algorithms.configs.transforms.CenterCropConfig.html#torchrl.trainers.algorithms.configs.transforms.CenterCropConfig)([height, width, in_keys, ...]) | Configuration for CenterCrop transform. |
| [`CropConfig`](generated/torchrl.trainers.algorithms.configs.transforms.CropConfig.html#torchrl.trainers.algorithms.configs.transforms.CropConfig)([top, left, height, width, ...]) | Configuration for Crop transform. |
| [`FlattenObservationConfig`](generated/torchrl.trainers.algorithms.configs.transforms.FlattenObservationConfig.html#torchrl.trainers.algorithms.configs.transforms.FlattenObservationConfig)([in_keys, ...]) | Configuration for FlattenObservation transform. |
| [`GrayScaleConfig`](generated/torchrl.trainers.algorithms.configs.transforms.GrayScaleConfig.html#torchrl.trainers.algorithms.configs.transforms.GrayScaleConfig)([in_keys, out_keys, _target_]) | Configuration for GrayScale transform. |
| [`ObservationNormConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ObservationNormConfig.html#torchrl.trainers.algorithms.configs.transforms.ObservationNormConfig)([loc, scale, in_keys, ...]) | Configuration for ObservationNorm transform. |
| [`CatFramesConfig`](generated/torchrl.trainers.algorithms.configs.transforms.CatFramesConfig.html#torchrl.trainers.algorithms.configs.transforms.CatFramesConfig)([N, dim, in_keys, out_keys, ...]) | Configuration for CatFrames transform. |
| [`RewardClippingConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RewardClippingConfig.html#torchrl.trainers.algorithms.configs.transforms.RewardClippingConfig)([clamp_min, clamp_max, ...]) | Configuration for RewardClipping transform. |
| [`RewardScalingConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RewardScalingConfig.html#torchrl.trainers.algorithms.configs.transforms.RewardScalingConfig)([loc, scale, in_keys, ...]) | Configuration for RewardScaling transform. |
| [`BinarizeRewardConfig`](generated/torchrl.trainers.algorithms.configs.transforms.BinarizeRewardConfig.html#torchrl.trainers.algorithms.configs.transforms.BinarizeRewardConfig)([in_keys, out_keys, ...]) | Configuration for BinarizeReward transform. |
| [`TargetReturnConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TargetReturnConfig.html#torchrl.trainers.algorithms.configs.transforms.TargetReturnConfig)([target_return, mode, ...]) | Configuration for TargetReturn transform. |
| [`VecNormConfig`](generated/torchrl.trainers.algorithms.configs.transforms.VecNormConfig.html#torchrl.trainers.algorithms.configs.transforms.VecNormConfig)([in_keys, out_keys, decay, ...]) | Configuration for VecNorm transform. |
| [`FrameSkipTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.FrameSkipTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.FrameSkipTransformConfig)([frame_skip, ...]) | Configuration for FrameSkipTransform. |
| [`DeviceCastTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.DeviceCastTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.DeviceCastTransformConfig)([device, in_keys, ...]) | Configuration for DeviceCastTransform. |
| [`DTypeCastTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.DTypeCastTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.DTypeCastTransformConfig)([dtype, in_keys, ...]) | Configuration for DTypeCastTransform. |
| [`UnsqueezeTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.UnsqueezeTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.UnsqueezeTransformConfig)([dim, in_keys, ...]) | Configuration for UnsqueezeTransform. |
| [`SqueezeTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.SqueezeTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.SqueezeTransformConfig)([dim, in_keys, ...]) | Configuration for SqueezeTransform. |
| [`PermuteTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.PermuteTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.PermuteTransformConfig)([dims, in_keys, ...]) | Configuration for PermuteTransform. |
| [`CatTensorsConfig`](generated/torchrl.trainers.algorithms.configs.transforms.CatTensorsConfig.html#torchrl.trainers.algorithms.configs.transforms.CatTensorsConfig)([dim, in_keys, out_keys, ...]) | Configuration for CatTensors transform. |
| [`StackConfig`](generated/torchrl.trainers.algorithms.configs.transforms.StackConfig.html#torchrl.trainers.algorithms.configs.transforms.StackConfig)([dim, in_keys, out_keys, _target_]) | Configuration for Stack transform. |
| [`DiscreteActionProjectionConfig`](generated/torchrl.trainers.algorithms.configs.transforms.DiscreteActionProjectionConfig.html#torchrl.trainers.algorithms.configs.transforms.DiscreteActionProjectionConfig)([...]) | Configuration for DiscreteActionProjection transform. |
| [`TensorDictPrimerConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TensorDictPrimerConfig.html#torchrl.trainers.algorithms.configs.transforms.TensorDictPrimerConfig)([primer_spec, ...]) | Configuration for TensorDictPrimer transform. |
| [`PinMemoryTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.PinMemoryTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.PinMemoryTransformConfig)([in_keys, ...]) | Configuration for PinMemoryTransform. |
| [`RewardSumConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RewardSumConfig.html#torchrl.trainers.algorithms.configs.transforms.RewardSumConfig)([in_keys, out_keys, ...]) | Configuration for RewardSum transform. |
| [`ExcludeTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ExcludeTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.ExcludeTransformConfig)([exclude_keys, _target_]) | Configuration for ExcludeTransform. |
| [`SelectTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.SelectTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.SelectTransformConfig)([include_keys, _target_]) | Configuration for SelectTransform. |
| [`TimeMaxPoolConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TimeMaxPoolConfig.html#torchrl.trainers.algorithms.configs.transforms.TimeMaxPoolConfig)([dim, in_keys, out_keys, ...]) | Configuration for TimeMaxPool transform. |
| [`RandomCropTensorDictConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RandomCropTensorDictConfig.html#torchrl.trainers.algorithms.configs.transforms.RandomCropTensorDictConfig)([crop_size, ...]) | Configuration for RandomCropTensorDict transform. |
| [`InitTrackerConfig`](generated/torchrl.trainers.algorithms.configs.transforms.InitTrackerConfig.html#torchrl.trainers.algorithms.configs.transforms.InitTrackerConfig)([init_key, _target_]) | Configuration for InitTracker transform. |
| [`RenameTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RenameTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.RenameTransformConfig)([key_mapping, _target_]) | Configuration for RenameTransform. |
| [`Reward2GoTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.Reward2GoTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.Reward2GoTransformConfig)([gamma, in_keys, ...]) | Configuration for Reward2GoTransform. |
| [`ActionMaskConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ActionMaskConfig.html#torchrl.trainers.algorithms.configs.transforms.ActionMaskConfig)([mask_key, in_keys, ...]) | Configuration for ActionMask transform. |
| [`VecGymEnvTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.VecGymEnvTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.VecGymEnvTransformConfig)([in_keys, ...]) | Configuration for VecGymEnvTransform. |
| [`BurnInTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.BurnInTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.BurnInTransformConfig)([burn_in, in_keys, ...]) | Configuration for BurnInTransform. |
| [`SignTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.SignTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.SignTransformConfig)([in_keys, out_keys, ...]) | Configuration for SignTransform. |
| [`RemoveEmptySpecsConfig`](generated/torchrl.trainers.algorithms.configs.transforms.RemoveEmptySpecsConfig.html#torchrl.trainers.algorithms.configs.transforms.RemoveEmptySpecsConfig)([_target_]) | Configuration for RemoveEmptySpecs transform. |
| [`BatchSizeTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.BatchSizeTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.BatchSizeTransformConfig)([batch_size, ...]) | Configuration for BatchSizeTransform. |
| [`AutoResetTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.AutoResetTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.AutoResetTransformConfig)([replace, ...]) | Configuration for AutoResetTransform. |
| [`ActionDiscretizerConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ActionDiscretizerConfig.html#torchrl.trainers.algorithms.configs.transforms.ActionDiscretizerConfig)([num_intervals, ...]) | Configuration for ActionDiscretizer transform. |
| [`TrajCounterConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TrajCounterConfig.html#torchrl.trainers.algorithms.configs.transforms.TrajCounterConfig)([out_key, repeats, _target_]) | Configuration for TrajCounter transform. |
| [`LineariseRewardsConfig`](generated/torchrl.trainers.algorithms.configs.transforms.LineariseRewardsConfig.html#torchrl.trainers.algorithms.configs.transforms.LineariseRewardsConfig)([in_keys, out_keys, ...]) | Configuration for LineariseRewards transform. |
| [`ConditionalSkipConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ConditionalSkipConfig.html#torchrl.trainers.algorithms.configs.transforms.ConditionalSkipConfig)([cond, _target_]) | Configuration for ConditionalSkip transform. |
| [`MultiActionConfig`](generated/torchrl.trainers.algorithms.configs.transforms.MultiActionConfig.html#torchrl.trainers.algorithms.configs.transforms.MultiActionConfig)([dim, stack_rewards, ...]) | Configuration for MultiAction transform. |
| [`TimerConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TimerConfig.html#torchrl.trainers.algorithms.configs.transforms.TimerConfig)([out_keys, time_key, _target_]) | Configuration for Timer transform. |
| [`ConditionalPolicySwitchConfig`](generated/torchrl.trainers.algorithms.configs.transforms.ConditionalPolicySwitchConfig.html#torchrl.trainers.algorithms.configs.transforms.ConditionalPolicySwitchConfig)([policy, ...]) | Configuration for ConditionalPolicySwitch transform. |
| [`FiniteTensorDictCheckConfig`](generated/torchrl.trainers.algorithms.configs.transforms.FiniteTensorDictCheckConfig.html#torchrl.trainers.algorithms.configs.transforms.FiniteTensorDictCheckConfig)([in_keys, ...]) | Configuration for FiniteTensorDictCheck transform. |
| [`UnaryTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.UnaryTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.UnaryTransformConfig)([fn, in_keys, ...]) | Configuration for UnaryTransform. |
| [`HashConfig`](generated/torchrl.trainers.algorithms.configs.transforms.HashConfig.html#torchrl.trainers.algorithms.configs.transforms.HashConfig)([in_keys, out_keys, _target_]) | Configuration for Hash transform. |
| [`TokenizerConfig`](generated/torchrl.trainers.algorithms.configs.transforms.TokenizerConfig.html#torchrl.trainers.algorithms.configs.transforms.TokenizerConfig)([vocab_size, in_keys, ...]) | Configuration for Tokenizer transform. |
| [`EndOfLifeTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.EndOfLifeTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.EndOfLifeTransformConfig)([eol_key, ...]) | Configuration for EndOfLifeTransform. |
| [`MultiStepTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.MultiStepTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.MultiStepTransformConfig)([n_steps, gamma, ...]) | Configuration for MultiStepTransform. |
| [`KLRewardTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.KLRewardTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.KLRewardTransformConfig)([in_keys, out_keys, ...]) | Configuration for KLRewardTransform. |
| [`R3MTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.R3MTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.R3MTransformConfig)([in_keys, out_keys, ...]) | Configuration for R3MTransform. |
| [`VC1TransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.VC1TransformConfig.html#torchrl.trainers.algorithms.configs.transforms.VC1TransformConfig)([in_keys, out_keys, ...]) | Configuration for VC1Transform. |
| [`VIPTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.VIPTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.VIPTransformConfig)([in_keys, out_keys, ...]) | Configuration for VIPTransform. |
| [`VIPRewardTransformConfig`](generated/torchrl.trainers.algorithms.configs.transforms.VIPRewardTransformConfig.html#torchrl.trainers.algorithms.configs.transforms.VIPRewardTransformConfig)([in_keys, ...]) | Configuration for VIPRewardTransform. |
| [`VecNormV2Config`](generated/torchrl.trainers.algorithms.configs.transforms.VecNormV2Config.html#torchrl.trainers.algorithms.configs.transforms.VecNormV2Config)([in_keys, out_keys, decay, ...]) | Configuration for VecNormV2 transform. |

### Data Collection Configurations

| [`CollectorConfig`](generated/torchrl.trainers.algorithms.configs.collectors.CollectorConfig.html#torchrl.trainers.algorithms.configs.collectors.CollectorConfig)([create_env_fn, policy, ...]) | Hydra configuration for [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector). |
| --- | --- |
| [`AsyncCollectorConfig`](generated/torchrl.trainers.algorithms.configs.collectors.AsyncCollectorConfig.html#torchrl.trainers.algorithms.configs.collectors.AsyncCollectorConfig)(create_env_fn, policy, ...) | Hydra configuration for [`AsyncCollector`](generated/torchrl.collectors.AsyncCollector.html#torchrl.collectors.AsyncCollector). |
| [`MultiSyncCollectorConfig`](generated/torchrl.trainers.algorithms.configs.collectors.MultiSyncCollectorConfig.html#torchrl.trainers.algorithms.configs.collectors.MultiSyncCollectorConfig)([create_env_fn, ...]) | Hydra configuration for [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector). |
| [`MultiAsyncCollectorConfig`](generated/torchrl.trainers.algorithms.configs.collectors.MultiAsyncCollectorConfig.html#torchrl.trainers.algorithms.configs.collectors.MultiAsyncCollectorConfig)([create_env_fn, ...]) | Hydra configuration for [`MultiAsyncCollector`](generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector). |

### Replay Buffer and Storage Configurations

| [`ReplayBufferConfig`](generated/torchrl.trainers.algorithms.configs.data.ReplayBufferConfig.html#torchrl.trainers.algorithms.configs.data.ReplayBufferConfig)([_partial_, _target_, ...]) | Hydra configuration for `ReplayBuffer`. |
| --- | --- |
| [`TensorDictReplayBufferConfig`](generated/torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig.html#torchrl.trainers.algorithms.configs.data.TensorDictReplayBufferConfig)([_partial_, ...]) | Hydra configuration for `TensorDictReplayBuffer`. |
| [`RandomSamplerConfig`](generated/torchrl.trainers.algorithms.configs.data.RandomSamplerConfig.html#torchrl.trainers.algorithms.configs.data.RandomSamplerConfig)([_target_]) | Configuration for random sampling from replay buffer. |
| [`SamplerWithoutReplacementConfig`](generated/torchrl.trainers.algorithms.configs.data.SamplerWithoutReplacementConfig.html#torchrl.trainers.algorithms.configs.data.SamplerWithoutReplacementConfig)([_target_, ...]) | Configuration for sampling without replacement. |
| [`PrioritizedSamplerConfig`](generated/torchrl.trainers.algorithms.configs.data.PrioritizedSamplerConfig.html#torchrl.trainers.algorithms.configs.data.PrioritizedSamplerConfig)([_target_, ...]) | Configuration for prioritized sampling from replay buffer. |
| [`SliceSamplerConfig`](generated/torchrl.trainers.algorithms.configs.data.SliceSamplerConfig.html#torchrl.trainers.algorithms.configs.data.SliceSamplerConfig)([_target_, num_slices, ...]) | Configuration for slice sampling from replay buffer. |
| [`SliceSamplerWithoutReplacementConfig`](generated/torchrl.trainers.algorithms.configs.data.SliceSamplerWithoutReplacementConfig.html#torchrl.trainers.algorithms.configs.data.SliceSamplerWithoutReplacementConfig)([...]) | Configuration for slice sampling without replacement. |
| [`ListStorageConfig`](generated/torchrl.trainers.algorithms.configs.data.ListStorageConfig.html#torchrl.trainers.algorithms.configs.data.ListStorageConfig)([_partial_, _target_, ...]) | Hydra configuration for [`ListStorage`](generated/torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage). |
| [`TensorStorageConfig`](generated/torchrl.trainers.algorithms.configs.data.TensorStorageConfig.html#torchrl.trainers.algorithms.configs.data.TensorStorageConfig)([_partial_, _target_, ...]) | Configuration for tensor-based storage in replay buffer. |
| [`LazyTensorStorageConfig`](generated/torchrl.trainers.algorithms.configs.data.LazyTensorStorageConfig.html#torchrl.trainers.algorithms.configs.data.LazyTensorStorageConfig)([_partial_, ...]) | Hydra configuration for [`LazyTensorStorage`](generated/torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage). |
| [`LazyMemmapStorageConfig`](generated/torchrl.trainers.algorithms.configs.data.LazyMemmapStorageConfig.html#torchrl.trainers.algorithms.configs.data.LazyMemmapStorageConfig)([_partial_, ...]) | Hydra configuration for [`LazyMemmapStorage`](generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage). |
| [`LazyStackStorageConfig`](generated/torchrl.trainers.algorithms.configs.data.LazyStackStorageConfig.html#torchrl.trainers.algorithms.configs.data.LazyStackStorageConfig)([_partial_, ...]) | Configuration for lazy stack storage. |
| [`StorageEnsembleConfig`](generated/torchrl.trainers.algorithms.configs.data.StorageEnsembleConfig.html#torchrl.trainers.algorithms.configs.data.StorageEnsembleConfig)([_partial_, _target_, ...]) | Configuration for storage ensemble. |
| [`RoundRobinWriterConfig`](generated/torchrl.trainers.algorithms.configs.data.RoundRobinWriterConfig.html#torchrl.trainers.algorithms.configs.data.RoundRobinWriterConfig)([_target_, compilable]) | Configuration for round-robin writer that distributes data across multiple storages. |
| [`StorageEnsembleWriterConfig`](generated/torchrl.trainers.algorithms.configs.data.StorageEnsembleWriterConfig.html#torchrl.trainers.algorithms.configs.data.StorageEnsembleWriterConfig)([_partial_, ...]) | Configuration for storage ensemble writer. |

### Training and Optimization Configurations

| [`TrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.TrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.TrainerConfig)() | Base configuration class for trainers. |
| --- | --- |
| [`PPOTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.PPOTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.PPOTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`PPOTrainer`](generated/torchrl.trainers.algorithms.PPOTrainer.html#torchrl.trainers.algorithms.PPOTrainer). |
| [`SACTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.SACTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.SACTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`SACTrainer`](generated/torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer). |
| [`DQNTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.DQNTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.DQNTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`DQNTrainer`](generated/torchrl.trainers.algorithms.DQNTrainer.html#torchrl.trainers.algorithms.DQNTrainer). |
| [`DDPGTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.DDPGTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.DDPGTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`DDPGTrainer`](generated/torchrl.trainers.algorithms.DDPGTrainer.html#torchrl.trainers.algorithms.DDPGTrainer). |
| [`IQLTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.IQLTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.IQLTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`IQLTrainer`](generated/torchrl.trainers.algorithms.IQLTrainer.html#torchrl.trainers.algorithms.IQLTrainer). |
| [`CQLTrainerConfig`](generated/torchrl.trainers.algorithms.configs.trainers.CQLTrainerConfig.html#torchrl.trainers.algorithms.configs.trainers.CQLTrainerConfig)(collector, total_frames, ...) | Hydra configuration for [`CQLTrainer`](generated/torchrl.trainers.algorithms.CQLTrainer.html#torchrl.trainers.algorithms.CQLTrainer). |

### Trainer Hook Configurations

| [`HookConfig`](generated/torchrl.trainers.algorithms.configs.hooks.HookConfig.html#torchrl.trainers.algorithms.configs.hooks.HookConfig)() | Base configuration class for trainer hooks. |
| --- | --- |
| [`BatchSubSamplerConfig`](generated/torchrl.trainers.algorithms.configs.hooks.BatchSubSamplerConfig.html#torchrl.trainers.algorithms.configs.hooks.BatchSubSamplerConfig)([batch_size, ...]) | Configuration for the [`BatchSubSampler`](generated/torchrl.trainers.BatchSubSampler.html#torchrl.trainers.BatchSubSampler) hook. |
| [`ClearCudaCacheConfig`](generated/torchrl.trainers.algorithms.configs.hooks.ClearCudaCacheConfig.html#torchrl.trainers.algorithms.configs.hooks.ClearCudaCacheConfig)([interval, _target_]) | Configuration for the [`ClearCudaCache`](generated/torchrl.trainers.ClearCudaCache.html#torchrl.trainers.ClearCudaCache) hook. |
| [`CountFramesLogConfig`](generated/torchrl.trainers.algorithms.configs.hooks.CountFramesLogConfig.html#torchrl.trainers.algorithms.configs.hooks.CountFramesLogConfig)([frame_skip, log_pbar, ...]) | Configuration for the [`CountFramesLog`](generated/torchrl.trainers.CountFramesLog.html#torchrl.trainers.CountFramesLog) hook. |
| [`EarlyStoppingConfig`](generated/torchrl.trainers.algorithms.configs.hooks.EarlyStoppingConfig.html#torchrl.trainers.algorithms.configs.hooks.EarlyStoppingConfig)([monitor, mode, ...]) | Configuration for the [`EarlyStopping`](generated/torchrl.trainers.EarlyStopping.html#torchrl.trainers.EarlyStopping) hook. |
| [`LogScalarConfig`](generated/torchrl.trainers.algorithms.configs.hooks.LogScalarConfig.html#torchrl.trainers.algorithms.configs.hooks.LogScalarConfig)([key, logname, log_pbar, ...]) | Configuration for the [`LogScalar`](generated/torchrl.trainers.LogScalar.html#torchrl.trainers.LogScalar) hook. |
| [`LogTimingConfig`](generated/torchrl.trainers.algorithms.configs.hooks.LogTimingConfig.html#torchrl.trainers.algorithms.configs.hooks.LogTimingConfig)([prefix, percall, erase, ...]) | Configuration for the `LogTiming` hook. |
| [`RewardNormalizerConfig`](generated/torchrl.trainers.algorithms.configs.hooks.RewardNormalizerConfig.html#torchrl.trainers.algorithms.configs.hooks.RewardNormalizerConfig)([decay, scale, eps, ...]) | Configuration for the [`RewardNormalizer`](generated/torchrl.trainers.RewardNormalizer.html#torchrl.trainers.RewardNormalizer) hook. |
| [`SelectKeysConfig`](generated/torchrl.trainers.algorithms.configs.hooks.SelectKeysConfig.html#torchrl.trainers.algorithms.configs.hooks.SelectKeysConfig)(keys, _target_) | Configuration for the [`SelectKeys`](generated/torchrl.trainers.SelectKeys.html#torchrl.trainers.SelectKeys) hook. |

| [`LossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.LossConfig.html#torchrl.trainers.algorithms.configs.objectives.LossConfig)([_partial_]) | A class to configure a loss. |
| --- | --- |
| [`PPOLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.PPOLossConfig.html#torchrl.trainers.algorithms.configs.objectives.PPOLossConfig)([_partial_, actor_network, ...]) | Hydra configuration for the PPO loss family. |
| [`SACLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.SACLossConfig.html#torchrl.trainers.algorithms.configs.objectives.SACLossConfig)([_partial_, actor_network, ...]) | Hydra configuration for [`SACLoss`](generated/torchrl.objectives.SACLoss.html#torchrl.objectives.SACLoss) (and [`DiscreteSACLoss`](generated/torchrl.objectives.DiscreteSACLoss.html#torchrl.objectives.DiscreteSACLoss) when `discrete=True`). |
| [`DQNLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.DQNLossConfig.html#torchrl.trainers.algorithms.configs.objectives.DQNLossConfig)([_partial_, value_network, ...]) | Hydra configuration for [`DQNLoss`](generated/torchrl.objectives.DQNLoss.html#torchrl.objectives.DQNLoss). |
| [`DDPGLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.DDPGLossConfig.html#torchrl.trainers.algorithms.configs.objectives.DDPGLossConfig)([_partial_, actor_network, ...]) | Hydra configuration for [`DDPGLoss`](generated/torchrl.objectives.DDPGLoss.html#torchrl.objectives.DDPGLoss). |
| [`IQLLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.IQLLossConfig.html#torchrl.trainers.algorithms.configs.objectives.IQLLossConfig)([_partial_, actor_network, ...]) | Hydra configuration for [`IQLLoss`](generated/torchrl.objectives.IQLLoss.html#torchrl.objectives.IQLLoss) (and [`DiscreteIQLLoss`](generated/torchrl.objectives.DiscreteIQLLoss.html#torchrl.objectives.DiscreteIQLLoss) when `discrete=True`). |
| [`CQLLossConfig`](generated/torchrl.trainers.algorithms.configs.objectives.CQLLossConfig.html#torchrl.trainers.algorithms.configs.objectives.CQLLossConfig)([_partial_, actor_network, ...]) | Hydra configuration for [`CQLLoss`](generated/torchrl.objectives.CQLLoss.html#torchrl.objectives.CQLLoss). |
| [`GAEConfig`](generated/torchrl.trainers.algorithms.configs.objectives.GAEConfig.html#torchrl.trainers.algorithms.configs.objectives.GAEConfig)([_partial_, gamma, lmbda, ...]) | Hydra configuration for [`GAE`](generated/torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE). |
| [`TargetNetUpdaterConfig`](generated/torchrl.trainers.algorithms.configs.objectives.TargetNetUpdaterConfig.html#torchrl.trainers.algorithms.configs.objectives.TargetNetUpdaterConfig)(loss_module[, _partial_]) | An abstract class to configure target net updaters. |
| [`SoftUpdateConfig`](generated/torchrl.trainers.algorithms.configs.objectives.SoftUpdateConfig.html#torchrl.trainers.algorithms.configs.objectives.SoftUpdateConfig)(loss_module[, _partial_, ...]) | A class for soft update instantiation. |
| [`HardUpdateConfig`](generated/torchrl.trainers.algorithms.configs.objectives.HardUpdateConfig.html#torchrl.trainers.algorithms.configs.objectives.HardUpdateConfig)(loss_module[, _partial_, ...]) | A class for hard update instantiation. |

| [`AdamConfig`](generated/torchrl.trainers.algorithms.configs.utils.AdamConfig.html#torchrl.trainers.algorithms.configs.utils.AdamConfig)([lr, betas, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam). |
| --- | --- |
| [`AdamWConfig`](generated/torchrl.trainers.algorithms.configs.utils.AdamWConfig.html#torchrl.trainers.algorithms.configs.utils.AdamWConfig)([lr, betas, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW). |
| [`AdamaxConfig`](generated/torchrl.trainers.algorithms.configs.utils.AdamaxConfig.html#torchrl.trainers.algorithms.configs.utils.AdamaxConfig)([lr, betas, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.Adamax`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax). |
| [`AdadeltaConfig`](generated/torchrl.trainers.algorithms.configs.utils.AdadeltaConfig.html#torchrl.trainers.algorithms.configs.utils.AdadeltaConfig)([lr, rho, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.Adadelta`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta). |
| [`AdagradConfig`](generated/torchrl.trainers.algorithms.configs.utils.AdagradConfig.html#torchrl.trainers.algorithms.configs.utils.AdagradConfig)([lr, lr_decay, weight_decay, ...]) | Hydra configuration for [`torch.optim.Adagrad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad). |
| [`ASGDConfig`](generated/torchrl.trainers.algorithms.configs.utils.ASGDConfig.html#torchrl.trainers.algorithms.configs.utils.ASGDConfig)([lr, lambd, alpha, t0, ...]) | Hydra configuration for [`torch.optim.ASGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD). |
| [`LBFGSConfig`](generated/torchrl.trainers.algorithms.configs.utils.LBFGSConfig.html#torchrl.trainers.algorithms.configs.utils.LBFGSConfig)([lr, max_iter, max_eval, ...]) | Configuration for LBFGS optimizer. |
| [`LionConfig`](generated/torchrl.trainers.algorithms.configs.utils.LionConfig.html#torchrl.trainers.algorithms.configs.utils.LionConfig)([lr, betas, weight_decay, ...]) | Configuration for Lion optimizer. |
| [`NAdamConfig`](generated/torchrl.trainers.algorithms.configs.utils.NAdamConfig.html#torchrl.trainers.algorithms.configs.utils.NAdamConfig)([lr, betas, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.NAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam). |
| [`RAdamConfig`](generated/torchrl.trainers.algorithms.configs.utils.RAdamConfig.html#torchrl.trainers.algorithms.configs.utils.RAdamConfig)([lr, betas, eps, weight_decay, ...]) | Hydra configuration for [`torch.optim.RAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam). |
| [`RMSpropConfig`](generated/torchrl.trainers.algorithms.configs.utils.RMSpropConfig.html#torchrl.trainers.algorithms.configs.utils.RMSpropConfig)([lr, alpha, eps, ...]) | Hydra configuration for [`torch.optim.RMSprop`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop). |
| [`RpropConfig`](generated/torchrl.trainers.algorithms.configs.utils.RpropConfig.html#torchrl.trainers.algorithms.configs.utils.RpropConfig)([lr, etas, step_sizes, ...]) | Hydra configuration for [`torch.optim.Rprop`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop). |
| [`SGDConfig`](generated/torchrl.trainers.algorithms.configs.utils.SGDConfig.html#torchrl.trainers.algorithms.configs.utils.SGDConfig)([lr, momentum, dampening, ...]) | Hydra configuration for [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD). |
| [`SparseAdamConfig`](generated/torchrl.trainers.algorithms.configs.utils.SparseAdamConfig.html#torchrl.trainers.algorithms.configs.utils.SparseAdamConfig)([lr, betas, eps, maximize, ...]) | Hydra configuration for [`torch.optim.SparseAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam). |

### Logging Configurations

| [`LoggerConfig`](generated/torchrl.trainers.algorithms.configs.logging.LoggerConfig.html#torchrl.trainers.algorithms.configs.logging.LoggerConfig)() | A class to configure a logger. |
| --- | --- |
| [`WandbLoggerConfig`](generated/torchrl.trainers.algorithms.configs.logging.WandbLoggerConfig.html#torchrl.trainers.algorithms.configs.logging.WandbLoggerConfig)(exp_name, offline, ...) | A class to configure a Wandb logger. |
| [`TensorboardLoggerConfig`](generated/torchrl.trainers.algorithms.configs.logging.TensorboardLoggerConfig.html#torchrl.trainers.algorithms.configs.logging.TensorboardLoggerConfig)(exp_name[, log_dir, ...]) | A class to configure a Tensorboard logger. |
| [`TrackioLoggerConfig`](generated/torchrl.trainers.algorithms.configs.logging.TrackioLoggerConfig.html#torchrl.trainers.algorithms.configs.logging.TrackioLoggerConfig)(exp_name, project, ...) | A class to configure a Trackio logger. |
| [`CSVLoggerConfig`](generated/torchrl.trainers.algorithms.configs.logging.CSVLoggerConfig.html#torchrl.trainers.algorithms.configs.logging.CSVLoggerConfig)(exp_name[, log_dir, ...]) | A class to configure a CSV logger. |

## Creating Custom Configurations

You can create custom configuration classes by inheriting from the appropriate base classes:

```
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
```

## Best Practices

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Use Defaults**: Leverage the `defaults` section to compose configurations
3. **Override Sparingly**: Only override what you need to change
4. **Validate Configurations**: Test that your configurations instantiate correctly
5. **Version Control**: Keep your configuration files under version control
6. **Use Variable Interpolation**: Use `${variable}` syntax to avoid duplication

## Supported Algorithms

TorchRL currently provides configuration-driven trainers for the following algorithms:

- **PPO** (on-policy): `PPOTrainerConfig`, `PPOLossConfig`
- **SAC** (off-policy, continuous): `SACTrainerConfig`, `SACLossConfig`
- **DQN** (off-policy, discrete): `DQNTrainerConfig`, `DQNLossConfig`
- **DDPG** (off-policy, continuous): `DDPGTrainerConfig`, `DDPGLossConfig`
- **IQL** (offline): `IQLTrainerConfig`, `IQLLossConfig`
- **CQL** (offline): `CQLTrainerConfig`, `CQLLossConfig`

The modular design ensures easy integration of additional algorithms while maintaining backward compatibility.