# RNDTransform

*class*torchrl.envs.transforms.RNDTransform(*target_network: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *predictor_network: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *in_keys: list[NestedKey] | None = None*, *out_keys: list[NestedKey] | None = None*, *normalize_obs: bool = True*, *normalize_reward: bool = True*, *obs_clip: float = 5.0*, *reward_clip: float = 5.0*)[[source]](../../_modules/torchrl/envs/transforms/rnd.html#RNDTransform)

Random Network Distillation transform that computes an intrinsic reward.

Implements the exploration bonus from:

> Burda et al., "Exploration by Random Network Distillation" (2018).
> [https://arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894)

At every environment step the transform:

1. Optionally normalizes the next observation with online running statistics
and clips the result to `[-obs_clip, obs_clip]` sigma.
2. Passes the (normalized) observation through both the frozen *target* and
the trainable *predictor* networks.
3. Writes the MSE prediction error as an intrinsic reward under `out_keys[0]`.
4. Optionally normalizes that reward by its running standard deviation.

The predictor is **only** given gradient updates through `RNDLoss`
during training. The transform itself always runs under `torch.no_grad()`.

Running normalization statistics are lazily initialized on the first step so
that the feature dimensionality does not need to be specified up-front. Pass
`normalize_obs=False` to skip observation normalization (useful when the
observation is already normalized by another transform).

Parameters:

- **target_network** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - frozen random network providing fixed
embeddings. Its parameters are frozen on construction.
- **predictor_network** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - trainable network that learns to
predict target embeddings.
- **in_keys** (*list**of**NestedKey**,**optional*) - tensordict keys to read
observations from. Defaults to `["observation"]`.
- **out_keys** (*list**of**NestedKey**,**optional*) - tensordict keys to write the
intrinsic reward to. Defaults to `["intrinsic_reward"]`.
- **normalize_obs** (*bool**,**optional*) - normalize observations with running
mean/std before passing to the networks. Default: `True`.
- **normalize_reward** (*bool**,**optional*) - divide intrinsic reward by its
running standard deviation. Default: `True`.
- **obs_clip** (*float**,**optional*) - clip normalized observations to
`[-obs_clip, obs_clip]`. Default: `5.0`.
- **reward_clip** (*float**,**optional*) - clip normalized intrinsic reward to
`[-reward_clip, reward_clip]`. Default: `5.0`.

Examples

```
>>> import torch.nn as nn
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> from torchrl.envs.transforms import RNDTransform
>>> target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
>>> predictor = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
>>> env = TransformedEnv(GymEnv("CartPole-v1"), RNDTransform(target, predictor))
>>> td = env.rollout(3)
>>> td["next", "intrinsic_reward"].shape
torch.Size([3, 1])
```

*property*obs_rms*: [RunningMeanStd](torchrl.envs.transforms.RunningMeanStd.html#torchrl.envs.transforms.RunningMeanStd) | None*

Running obs statistics, or `None` before the first step.

*property*reward_rms*: [RunningMeanStd](torchrl.envs.transforms.RunningMeanStd.html#torchrl.envs.transforms.RunningMeanStd) | None*

Running intrinsic-reward statistics, or `None` before the first step.

transform_reward_spec(*reward_spec*)[[source]](../../_modules/torchrl/envs/transforms/rnd.html#RNDTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform