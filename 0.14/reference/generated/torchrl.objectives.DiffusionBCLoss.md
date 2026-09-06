# DiffusionBCLoss

*class*torchrl.objectives.DiffusionBCLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/diffusion_bc.html#DiffusionBCLoss)

Behavioural Cloning loss for diffusion-based policies.

Implements the ε-prediction (noise-prediction) denoising loss from
[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) (Chi et al., RSS 2023).

Given a batch of (observation, clean_action) pairs from a demonstration
dataset, the loss:

1. Samples a random diffusion timestep `t` for each item in the batch.
2. Corrupts the clean action with Gaussian noise via the DDPM forward
process: `noisy_action = sqrt(ᾱ_t) * action + sqrt(1 - ᾱ_t) * ε`.
3. Asks the score network to predict the noise `ε`.
4. Returns the MSE between the predicted and actual noise.

This loss is designed to be used together with
`DiffusionActor`. The actor's inner
`_DDPMModule` is
accessed via `actor_network.module` and its `add_noise` method is
used for step 2.

Parameters:

**actor_network** (*TensorDictModule*) - a `DiffusionActor`
(or any [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) whose `.module`
exposes `add_noise(clean_action, t)` and a
`score_network` attribute).

Keyword Arguments:

**reduction** (*str**,**optional*) - Specifies the reduction to apply to the
output: `"none"` | `"mean"` | `"sum"`. Defaults to
`"mean"`.

Note

The tensordict passed to `forward()` must contain:

- `self.tensor_keys.action` -- the *clean* (demonstration) action.
- `self.tensor_keys.observation` -- the conditioning observation.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import DiffusionActor
>>> from torchrl.objectives import DiffusionBCLoss
>>> actor = DiffusionActor(action_dim=2, obs_dim=4, num_steps=10)
>>> loss_fn = DiffusionBCLoss(actor)
>>> td = TensorDict(
... {
... "observation": torch.randn(8, 4),
... "action": torch.randn(8, 2),
... },
... batch_size=[8],
... )
>>> loss_td = loss_fn(td)
>>> loss_td["loss_diffusion_bc"].backward()
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/diffusion_bc.html#DiffusionBCLoss.forward)

Compute the diffusion BC loss.

Parameters:

**tensordict** (*TensorDictBase*) - input data containing observations
and clean demonstration actions.

Returns:

TensorDict with key `"loss_diffusion_bc"`.