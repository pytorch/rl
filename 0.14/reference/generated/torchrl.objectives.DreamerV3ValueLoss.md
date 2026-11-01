# DreamerV3ValueLoss

*class*torchrl.objectives.DreamerV3ValueLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ValueLoss)

DreamerV3 Value Loss.

See [DreamerV3 in a nutshell](../dreamer_v3.html) for an overview
of the online critic, slow critic, and their update flow.

Trains the value network to predict the lambda-target computed by
[`DreamerV3ActorLoss`](torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss). Supports two loss modes:

- `"symlog_mse"` (default): `(symlog(v_pred) - symlog(target))^2`
- `"two_hot"`: Two-hot cross-entropy over a fixed bin grid (matches the
full DreamerV3 distribution-valued critic).

The discount factor used here must match the one the actor used to compute
`lambda_target`. The recommended way to keep them in lock-step is to
pass the actor loss to the constructor via `actor_loss=`: the value loss
will then read `gamma` from the actor's value estimator at every forward
call. The legacy `gamma=` kwarg + `sync_gamma_with_actor_loss()`
pattern is still supported.

Setting `slow_critic_regularization` to a positive value creates a
checkpointed target critic. Associate a `SoftUpdate`
and step it after each critic optimizer step.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **value_model** (*TensorDictModule*) - The value network.
- **value_loss** (*"symlog_mse"**or**"two_hot"**,**optional*) - Loss type.
Default: `"symlog_mse"`.
- **discount_loss** (*bool**,**optional*) - If `True`, discounts the loss with
a cumulative gamma factor. Default: `True`.
- **gamma** (*float**,**optional*) - Discount factor used when `discount_loss=True`.
Ignored if `actor_loss` is provided. Default: `0.99`.
- **num_value_bins** (*int**,**optional*) - Number of bins for `"two_hot"` loss.
Default: 255.
- **actor_loss** ([*DreamerV3ActorLoss*](torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss)*,**optional*) - If provided, `gamma` is
read from this actor loss's value estimator on every forward call,
avoiding any chance of a mismatch. Default: `None`.
- **slow_critic_regularization** (*float**,**optional*) - Weight of the auxiliary
loss that trains the online critic toward decoded target-critic
predictions. Default: `0.0`.
- **reduction** (*"none"**,**"mean"**or**"sum"**,**optional*) - Reduction applied to
the loss. Defaults to `"mean"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import MLP
>>> from torchrl.objectives import DreamerV3ValueLoss
>>> value_model = TensorDictModule(
... MLP(out_features=1, depth=1, num_cells=8),
... in_keys=["state"],
... out_keys=["state_value"],
... )
>>> td = TensorDict({
... "state": torch.randn(8, 4),
... "lambda_target": torch.randn(8, 1),
... }, [8])
>>> loss = DreamerV3ValueLoss(value_model)
>>> loss_td, _ = loss(td)
>>> "loss_value" in loss_td.keys()
True
```

default_keys

alias of `_AcceptedKeys`

forward(*fake_data*) → tuple[[TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)][[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ValueLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

replay_value_loss(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *horizon: float = 333.0*, *lmbda: float = 0.95*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ValueLoss.replay_value_loss)

Compute the DreamerV3 critic loss on a replay sequence.

The return of each replay state uses the reward of the next step and
bootstraps from `bootstrap`, the first imagined return of that
state. The gradient stays on the input features, so the loss can
train the RSSM representation when the model loss does not detach.

Parameters:

- **tensordict** (*TensorDictBase*) - Posterior replay features, batch size
`[B, T]`, with the `value_model` input keys and the
`reward`, `done`, `terminated` and `bootstrap` entries
that `tensor_keys` names.
- **horizon** (*float**,**optional*) - Discount horizon; the step discount is
`1 - 1 / horizon`. Defaults to `333.0`.
- **lmbda** (*float**,**optional*) - Lambda-return coefficient. Default: 0.95.

Returns:

A tensordict with the scalar, unweighted `loss_replay_value`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import MLP
>>> from torchrl.objectives import DreamerV3ValueLoss
>>> value_model = TensorDictModule(
... MLP(out_features=1, depth=1, num_cells=8),
... in_keys=["state"],
... out_keys=["state_value"],
... )
>>> loss = DreamerV3ValueLoss(value_model)
>>> replay = TensorDict({
... "state": torch.randn(2, 5, 4),
... "bootstrap": torch.randn(2, 5),
... "next": {
... "reward": torch.randn(2, 5, 1),
... "done": torch.zeros(2, 5, 1, dtype=torch.bool),
... "terminated": torch.zeros(2, 5, 1, dtype=torch.bool),
... },
... }, [2, 5])
>>> loss_td = loss.replay_value_loss(replay)
>>> loss_td["loss_replay_value"].shape
torch.Size([])
```

sync_gamma_with_actor_loss(*actor_loss: [DreamerV3ActorLoss](torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss)*) → None[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ValueLoss.sync_gamma_with_actor_loss)

Pull `gamma` from an actor loss's value estimator.

Prefer passing `actor_loss=` to the constructor; this method exists
for backward compatibility with the legacy two-step setup.