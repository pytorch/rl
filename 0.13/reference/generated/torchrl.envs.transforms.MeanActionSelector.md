# MeanActionSelector

*class*torchrl.envs.transforms.MeanActionSelector(*observation_key: str = 'observation'*, *action_key: str = 'action'*)[[source]](../../_modules/torchrl/envs/transforms/mean_action_selector.html#MeanActionSelector)

Bridges Gaussian belief-space policies with standard environments.

Gaussian policies used in moment-matching model-based RL (e.g. PILCO) operate
on state *beliefs* - `(mean, covariance)` pairs - and produce
action distributions with `("action", "mean")`, `("action", "var")`, etc.
This transform adapts a standard environment so that such a policy can be
used directly with [`rollout()`](torchrl.envs.EnvBase.html#id2):

- **Forward** (env output -> policy input): wraps the flat `"observation"`
tensor into `("observation", "mean")` with a zero-covariance
`("observation", "var")`, representing a deterministic state belief.
- **Inverse** (policy output -> env input): extracts `("action", "mean")`
from the policy output and writes it as the flat `"action"` for the
base environment step.

Parameters:

- **observation_key** (*str**,**optional*) - The observation key to read from the
base environment. Defaults to `"observation"`.
- **action_key** (*str**,**optional*) - The action key expected by the base
environment. Defaults to `"action"`.

Examples

```
>>> import torch
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> from torchrl.envs.transforms import MeanActionSelector
>>> base_env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(base_env, MeanActionSelector())
>>> td = env.reset()
>>> # The policy now sees ("observation", "mean") and ("observation", "var")
>>> print(td["observation", "mean"].shape)
>>> print(td["observation", "var"].shape)
```

transform_observation_spec(*observation_spec*)[[source]](../../_modules/torchrl/envs/transforms/mean_action_selector.html#MeanActionSelector.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform