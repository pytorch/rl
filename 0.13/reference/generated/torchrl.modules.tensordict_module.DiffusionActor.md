# DiffusionActor

*class*torchrl.modules.tensordict_module.DiffusionActor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DiffusionActor)

Diffusion-based actor for RL.

Implements a score-based policy that denoises latent actions conditioned on
observations using a fixed DDPM scheduler. A small MLP is used as the
score network by default; pass a custom `score_network` to override.

The strict TensorDict contract is `in_keys=["observation"]` →
`out_keys=["action"]`.

Respects `interaction_type()`: setting
the interaction type to `DETERMINISTIC` disables stochastic noise
injection during the reverse chain, yielding a deterministic output.

Parameters:

- **action_dim** (*int*) - Dimensionality of the action space.
- **obs_dim** (*int**,**optional*) - Dimensionality of the observation space.
Only required when `score_network` is `None` (i.e., when the
default MLP is used). When a custom `score_network` is
provided this argument is ignored. Defaults to `None`.
- **score_network** (*nn.Module**,**optional*) - Network that predicts noise given
`(noisy_action, observation, timestep)` concatenated along the
last dimension. If `None`, a two-hidden-layer MLP of width 256
with a [`LazyLinear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) first layer is constructed
automatically (`obs_dim` need not be specified in this case).
- **num_steps** (*int*) - Number of DDPM denoising steps. Defaults to 100.
- **beta_start** (*float*) - Starting beta for the linear schedule.
Defaults to 1e-4.
- **beta_end** (*float*) - Ending beta for the linear schedule.
Defaults to 0.02.
- **in_keys** (*list**of**NestedKey**,**optional*) - Keys read from the input
TensorDict. Defaults to `["observation"]`.
- **out_keys** (*list**of**NestedKey**,**optional*) - Keys written to the output
TensorDict. Defaults to `["action"]`.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - Spec for the action output.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import DiffusionActor
>>> # obs_dim not required when using the default network
>>> actor = DiffusionActor(action_dim=2, num_steps=10)
>>> td = TensorDict({"observation": torch.randn(4, 3)}, batch_size=[4])
>>> td = actor(td)
>>> td["action"].shape
torch.Size([4, 2])
```