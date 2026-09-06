# WorldModel

*class*torchrl.modules.WorldModel(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModel)

A general, composable world model for model-based RL.

`WorldModel` wraps an encoder, a dynamics model, a reward head, and
optionally a done head and a decoder into a single TensorDict-native
module. It owns *prediction and composition* -- encoding observations,
advancing latent state, predicting rewards and termination -- and exposes
named shortcuts (`encode()`, `step()`, `decode()`) so each
component can be invoked individually.

Rollout semantics live elsewhere: wrap a `WorldModel` in
`WorldModelEnv` (or another
`ModelBasedEnvBase` subclass) and use
[`rollout()`](torchrl.envs.EnvBase.html#id2) to generate imagined trajectories.
This keeps env-level concerns -- reset/step contract, `done` handling,
spec validation -- out of the prediction module and avoids forking a
second rollout implementation with subtly different semantics.

The module is key-driven: each component communicates through named
TensorDict keys defined by its `in_keys` / `out_keys`. No specific
latent representation, observation space, or dynamics architecture is
assumed.

Parameters:

- **encoder** (*TensorDictModule*) - maps an observation to a latent
representation, e.g. `obs -> latent`.
- **dynamics** (*TensorDictModule*) - advances the latent state given an
action, e.g. `(latent, action) -> ("next", latent)`.
- **reward_head** (*TensorDictModule*) - predicts the reward from the next
latent, e.g. `("next", latent) -> ("next", "reward")`.
- **done_head** (*TensorDictModule**,**optional*) - predicts the done flag, e.g.
`("next", latent) -> ("next", "done")`. When provided,
`rollout()` can terminate early when any trajectory is done.
- **decoder** (*TensorDictModule**,**optional*) - reconstructs an observation from
a latent, e.g. `latent -> obs_recon`. Required to call
`decode()`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import WorldModel
>>> obs_dim, latent_dim, action_dim = 8, 4, 2
>>> encoder = TensorDictModule(
... torch.nn.Linear(obs_dim, latent_dim),
... in_keys=["observation"],
... out_keys=["latent"],
... )
>>> dynamics = TensorDictModule(
... torch.nn.Linear(latent_dim + action_dim, latent_dim),
... in_keys=["latent", "action"],
... out_keys=[("next", "latent")],
... )
>>> reward_head = TensorDictModule(
... torch.nn.Linear(latent_dim, 1),
... in_keys=[("next", "latent")],
... out_keys=[("next", "reward")],
... )
>>> world_model = WorldModel(encoder, dynamics, reward_head)
>>> td = TensorDict(
... {"observation": torch.randn(2, obs_dim), "action": torch.randn(2, action_dim)},
... batch_size=[2],
... )
>>> out = world_model(td)
>>> out.keys()
dict_keys(['observation', 'action', 'latent', 'next'])
```

decode(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModel.decode)

Decode a latent back to observation space.

Raises:

**RuntimeError** - if no `decoder` was provided at construction.

encode(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModel.encode)

Encode an observation into the latent space.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModel.forward)

Run the full pipeline: encoder -> dynamics -> reward_head -> [done_head] -> [decoder].

step(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModel.step)

Take one imagined step: dynamics -> reward_head -> [done_head] -> [decoder].

The encoder is *not* called; the tensordict must already contain the
current latent state as produced by `encode()` or a previous call
to `step()`.

*property*step_module*: [TensorDictSequential](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)*

The step-only sequence (dynamics + heads, no encoder).

Exposed as a public attribute so `WorldModelEnv`
and other model-based env wrappers can drive the world model in latent
space, one step at a time, without rerunning the encoder on every step.