# SuccessReward

*class*torchrl.envs.transforms.SuccessReward(*success_key: NestedKey = 'success'*, *reward_key: NestedKey = 'reward'*, ***, *scale: float = 1.0*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#SuccessReward)

Sparse 0/1 success reward for reinforcement fine-tuning.

Reads a boolean (or 0/1) success signal and writes a sparse reward
(`scale` on success, `0` otherwise). This is the trajectory-level
success reward used by SimpleVLA-RL / RL4VLA-style VLA RL, where a binary
task-completion signal is the only reward, but it is a general transform:
sparse task-completion rewards are ubiquitous in goal-conditioned RL.

It is a standard leaf transform: it can be appended to a
`TransformedEnv` (it overwrites the step reward from the
env's success signal) or applied to sampled data in a replay buffer. When
attached to an environment, the reward spec is rewritten to a
[`Bounded`](torchrl.data.Bounded.html#torchrl.data.Bounded) spec over `{0, scale}` (shaped like the
success entry); the reward is written at step time only, never at reset.

Parameters:

- **success_key** (*NestedKey*) - the boolean success signal to read.
Defaults to `"success"`.
- **reward_key** (*NestedKey*) - the reward to write. Defaults to `"reward"`.

Keyword Arguments:

**scale** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)) - the reward value on success. Defaults to `1.0`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs.transforms import SuccessReward
>>> t = SuccessReward(scale=1.0)
>>> td = TensorDict({"success": torch.tensor([[True], [False]])}, batch_size=[2])
>>> t(td)["reward"].squeeze(-1).tolist()
[1.0, 0.0]
```

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#SuccessReward.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform