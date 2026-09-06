# WorldModelLoss

*class*torchrl.objectives.WorldModelLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/world_model_loss.html#WorldModelLoss)

A general loss module for model-based world models.

`WorldModelLoss` evaluates a [`WorldModel`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel) on a
batch of real transitions and returns a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
containing one or more named sub-losses. All sub-losses are optional and
controlled via the `losses` argument:

- `"reward"`: MSE / L1 between the predicted reward and the ground-truth
reward stored in the replay buffer.
- `"done"`: MSE / L1 between the predicted done flag and the ground-truth
done flag.
- `"reconstruction"`: MSE / L1 between the decoder's reconstructed
observation and the original observation.
- `"latent"`: MSE / L1 between a predicted next-latent key and a
target next-latent key. Useful for deterministic world models; for
VAE / RSSM-style KL losses use
[`DreamerModelLoss`](torchrl.objectives.DreamerModelLoss.html#torchrl.objectives.DreamerModelLoss) instead.

The ground-truth reward and done tensors are read from the input
tensordict, renamed to `("next", true_reward)` / `("next", true_done)`
before the world model is called, so that the world model can freely write
its predictions under `("next", reward)` / `("next", done)`.

Parameters:

- **world_model** ([*WorldModel*](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel)) - the world model to evaluate.
- **losses** (*list**of**str**,**optional*) - which sub-losses to compute.
Any subset of `["reward", "done", "reconstruction", "latent"]`.
Defaults to `["reward"]`.
- **reward_loss** (*str**,**optional*) - loss function for the reward head.
Passed to `distance_loss()`.
Default: `"l2"`.
- **done_loss** (*str**,**optional*) - loss function for the done head.
Default: `"l2"`.
- **reconstruction_loss** (*str**,**optional*) - loss function for the decoder.
Default: `"l2"`.
- **latent_loss** (*str**,**optional*) - loss function for the latent prediction.
Default: `"l2"`.
- **reward_weight** (*float**,**optional*) - scalar weight applied to
`loss_reward`. Default: `1.0`.
- **done_weight** (*float**,**optional*) - scalar weight applied to
`loss_done`. Default: `1.0`.
- **reconstruction_weight** (*float**,**optional*) - scalar weight applied to
`loss_reconstruction`. Default: `1.0`.
- **latent_weight** (*float**,**optional*) - scalar weight applied to
`loss_latent`. Default: `1.0`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import WorldModel
>>> from torchrl.objectives import WorldModelLoss
>>> obs_dim, latent_dim, action_dim = 8, 4, 2
>>> encoder = TensorDictModule(
... torch.nn.Linear(obs_dim, latent_dim),
... in_keys=["observation"], out_keys=["latent"],
... )
>>> dynamics = TensorDictModule(
... torch.nn.Linear(latent_dim + action_dim, latent_dim),
... in_keys=["latent", "action"], out_keys=[("next", "latent")],
... )
>>> reward_head = TensorDictModule(
... torch.nn.Linear(latent_dim, 1),
... in_keys=[("next", "latent")], out_keys=[("next", "reward")],
... )
>>> world_model = WorldModel(encoder, dynamics, reward_head)
>>> loss_module = WorldModelLoss(world_model, losses=["reward"])
>>> batch = TensorDict(
... {
... "observation": torch.randn(4, obs_dim),
... "action": torch.randn(4, action_dim),
... "next": {"reward": torch.randn(4, 1)},
... },
... batch_size=[4],
... )
>>> loss_td = loss_module(batch)
>>> loss_td.keys()
dict_keys(['loss_reward'])
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/world_model_loss.html#WorldModelLoss.forward)

Compute the world model loss on a batch of real transitions.

Parameters:

**tensordict** (*TensorDictBase*) - a batch of real transitions containing
at minimum the keys consumed by the world model.

Returns:

a scalar TensorDict with keys `"loss_reward"`,
`"loss_done"`, `"loss_reconstruction"`, and/or
`"loss_latent"` depending on the active `losses`.

Return type:

TensorDictBase