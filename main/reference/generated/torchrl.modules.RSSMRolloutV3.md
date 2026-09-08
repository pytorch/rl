# RSSMRolloutV3

*class*torchrl.modules.RSSMRolloutV3(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMRolloutV3)

Roll out the DreamerV3 RSSM over a sequence.

See [DreamerV3 in a nutshell](../dreamer_v3.html) for the RSSM
data flow and terminology used by this rollout.

Given encoded observations and actions for `T` time steps, this module
runs the prior (GRU + categorical) then the posterior (categorical) at each
step and returns a stacked TensorDict of all intermediate states.

The previous posterior state `z_t` is used as the prior input for step
`t+1`, matching the recurrent structure of DreamerV3.

The module picks one of two paths at construction: tensors when the
modules use the standard DreamerV3 key wiring, TensorDicts otherwise. Both
give identical results, and the tensor path shares storage for the entries
it does not overwrite. See `compile_rollout()`.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **rssm_prior** (*TensorDictModule*) - Prior module wrapping [`RSSMPriorV3`](torchrl.modules.RSSMPriorV3.html#torchrl.modules.RSSMPriorV3).
- **rssm_posterior** (*TensorDictModule*) - Posterior module wrapping
[`RSSMPosteriorV3`](torchrl.modules.RSSMPosteriorV3.html#torchrl.modules.RSSMPosteriorV3).
- **reset_key** (*NestedKey**or**None**,**optional*) - Boolean key marking the first
transition of an episode. The rollout zeroes the state, belief and
action there. Defaults to `"is_init"`.
- **action_key** (*NestedKey**or**None**,**optional*) - Action key, zeroed on a reset
step. Defaults to `None`: the module then takes the
`rssm_prior` input key that is not `"state"` or `"belief"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.models.model_based import (
... RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3,
... )
>>> prior = TensorDictModule(
... RSSMPriorV3(action_shape=torch.Size([2]), hidden_dim=8,
... rnn_hidden_dim=8, num_categoricals=4, num_classes=4,
... action_dim=2),
... in_keys=["state", "belief", "action"],
... out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
... )
>>> posterior = TensorDictModule(
... RSSMPosteriorV3(hidden_dim=8, num_categoricals=4, num_classes=4,
... rnn_hidden_dim=8, obs_embed_dim=6),
... in_keys=[("next", "belief"), ("next", "encoded_latents")],
... out_keys=[("next", "posterior_logits"), ("next", "state")],
... )
>>> rollout = RSSMRolloutV3(prior, posterior)
>>> td = TensorDict({
... "state": torch.zeros(2, 4, 16),
... "belief": torch.zeros(2, 4, 8),
... "action": torch.randn(2, 4, 2),
... "next": {"encoded_latents": torch.randn(2, 4, 6)},
... }, [2, 4])
>>> out = rollout(td)
>>> out.shape
torch.Size([2, 4])
```

compile_rollout(*scope: Literal['step', 'scan'] = 'step'*, ***, *unroll: int = 1*, ***compile_kwargs*) → None[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMRolloutV3.compile_rollout)

Compile the recurrence with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

`"step"` compiles one deterministic step of the default explicit
loop. `"scan"` selects and compiles the higher-order scan backend.
Random samples are supplied as higher-order scan inputs. Eager and
compiled executions are not expected to consume identical RNG streams.

Both scopes need the tensor path.

Parameters:

- **scope** (*"step"**or**"scan"**,**optional*) - Part of the recurrence to
compile. Defaults to `"step"`.
- **unroll** (*int**,**optional*) - Number of scan steps to trace in each
higher-order scan iteration. Larger values can improve runtime
at the cost of compilation time and graph size. Only applies
to `scope="scan"`. Defaults to `1`.
- ****compile_kwargs** - Keyword arguments for [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
`dynamic` defaults to `False`.

forward(*tensordict*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMRolloutV3.forward)

Roll out the RSSM for one episode chunk.

Parameters:

**tensordict** (*TensorDictBase*) - Input with shape `[*batch, T]` containing
actions, encoded observations, and initial state/belief.

Returns:

Stacked outputs with shape `[*batch, T]`.

Return type:

TensorDictBase