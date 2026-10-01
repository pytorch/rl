# TdMpc2Loss

*class*torchrl.objectives.TdMpc2Loss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/tdmpc2.html#TdMpc2Loss)

Compute the TD-MPC2 model-learning objective.

Parameters:

- **world_model** - TensorDict-native encoder, dynamics, and reward model.
- **policy_prior** - TensorDict module producing sampled actions and policy
statistics from a latent state.
- **q_ensemble** - Distributional Q-function ensemble.
- **horizon** - Number of transitions in each sampled sequence.
- **discount** - Scalar discount factor used for TD targets.
- **rho** - Temporal weighting factor for model and actor losses.
- **consistency_coef** - Weight of latent consistency loss.
- **reward_coef** - Weight of distributional reward loss.
- **value_coef** - Weight of distributional value loss.
- **entropy_coef** - Entropy coefficient in the actor objective.
- **scale_tau** - Exponential averaging factor for the running value scale.
- **observation_key** - Current observation key.
- **action_key** - Action key.
- **reward_key** - Reward key under `"next"`.
- **terminated_key** - Termination key under `"next"`.

See also

`TdMpc2LossConfig`,
[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.objectives import TdMpc2Loss
>>> # The three TensorDict-native components are built from the
>>> # corresponding TD-MPC2 configuration classes.
>>> loss = TdMpc2Loss(world_model, policy_prior, q_ensemble)
```

actor_loss_from_latents(*latent_sequence: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/objectives/tdmpc2.html#TdMpc2Loss.actor_loss_from_latents)

Compute the actor objective from detached imagined latents.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/tdmpc2.html#TdMpc2Loss.forward)

Return the independent weighted model-loss components.

*property*in_keys*: list[NestedKey]*

Return the canonical current and next transition keys.

model_loss(*sample: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/objectives/tdmpc2.html#TdMpc2Loss.model_loss)

Compute the model, reward, value, and consistency objectives.

Parameters:

**sample** - Canonical batch-major transition sequence with final batch
dimension equal to `horizon`.

Returns:

The weighted model loss and detached latent/model metadata for the
subsequent actor phase.

*property*out_keys*: list[NestedKey]*

Return the scalar loss and diagnostic keys written by `forward`.