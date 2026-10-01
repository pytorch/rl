# RNDLoss

*class*torchrl.objectives.RNDLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/rnd.html#RNDLoss)

Loss module for training the predictor network in Random Network Distillation.

Presented in:

> Burda et al., "Exploration by Random Network Distillation" (2018).
> [https://arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894)

Computes the MSE between the *predictor* and the frozen *target* network on
next observations sampled from a replay buffer. Call this loss alongside
your main policy objective; its gradients update the predictor so that
familiar observations gradually yield lower intrinsic rewards.

The `predictor_network` and `target_network` should be the
**same objects** passed to [`RNDTransform`](torchrl.envs.transforms.RNDTransform.html#torchrl.envs.transforms.RNDTransform)
so that reducing the predictor error here also reduces the intrinsic reward
produced during collection.

Observation normalization is optionally applied using the running statistics
maintained by [`RNDTransform`](torchrl.envs.transforms.RNDTransform.html#torchrl.envs.transforms.RNDTransform). Pass
`obs_rms=transform.obs_rms` after collecting initial data to keep the
normalization consistent between collection and training.

Parameters:

- **predictor_network** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - trainable network.
- **target_network** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - frozen random network. Its parameters
are frozen on construction.

Keyword Arguments:

- **obs_rms** ([*RunningMeanStd*](torchrl.envs.transforms.RunningMeanStd.html#torchrl.envs.transforms.RunningMeanStd)*,**optional*) - running observation statistics
shared with [`RNDTransform`](torchrl.envs.transforms.RNDTransform.html#torchrl.envs.transforms.RNDTransform).
When provided, observations are normalized before being passed to
the networks, matching the normalization done during collection.
Defaults to `None` (no normalization).
- **obs_clip** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - clip normalized observations to
`[-obs_clip, obs_clip]`. Only used when `obs_rms` is not
`None`. Default: `5.0`.
- **reduction** (*str**,**optional*) - reduction over the per-sample losses:
`"mean"` | `"sum"` | `"none"`. Default: `"mean"`.
- **update_fraction** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - fraction of each batch used to
compute the predictor loss, following the original paper (default
25 %). A random mask selects which samples contribute so the
operation is `torch.compile`-friendly. Default: `0.25`.

Examples

```
>>> import torch
>>> import torch.nn as nn
>>> from tensordict import TensorDict
>>> from torchrl.objectives.rnd import RNDLoss
>>> predictor = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
>>> target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
>>> loss_fn = RNDLoss(predictor, target)
>>> batch = TensorDict({"next": {"observation": torch.randn(32, 4)}}, [32])
>>> loss_td = loss_fn(batch)
>>> loss_td["loss_predictor"].backward()
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/rnd.html#RNDLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.