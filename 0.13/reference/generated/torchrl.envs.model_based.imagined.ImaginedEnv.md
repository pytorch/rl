# ImaginedEnv

torchrl.envs.model_based.imagined.ImaginedEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/model_based/imagined.html#ImaginedEnv)

Imagination environment for model-based policy search.

Wraps a learned world model (e.g. a Gaussian Process) as a standard
TorchRL environment so that imagined rollouts can be collected with
[`rollout()`](torchrl.envs.EnvBase.html#id2). Observations carry both mean
and covariance (under keys `("observation", "mean")` and
`("observation", "var")`) to support uncertainty-aware moment-matching
controllers.

The environment never terminates on its own - rollout length is
controlled solely by the `max_steps` argument of
[`rollout()`](torchrl.envs.EnvBase.html#id2). The `done` and `terminated`
flags are always `False`.

Parameters:

- **world_model_module** (*TensorDictModule*) - A [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)
that takes `"action"` and `"observation"` entries and produces
`("next_observation", "mean")` and `("next_observation", "var")`.
- **base_env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - The real environment whose specs (observation, action,
reward, done) are copied into this imagined environment.
- **batch_size** (*int**,**Sequence**[**int**]**,*[*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - Override batch size.
If `None`, inferred from `base_env` (with a minimum of `[1]`).
- **next_observation_key** (*str**or**tuple**of**str**,**optional*) - The key where the world
model writes the predicted next observation. Defaults to `("next", "observation")`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.envs.model_based import ImaginedEnv, ModelBasedEnvBase
>>> from torchrl.data import Composite, Unbounded
>>> base_env = GymEnv("Pendulum-v1")
>>> obs_dim = base_env.observation_spec["observation"].shape[-1]
>>> # A toy world model that returns zero-mean, identity covariance
>>> class DummyWorldModel(torch.nn.Module):
... def __init__(self, obs_dim):
... super().__init__()
... self.obs_dim = obs_dim
... def forward(self, action, observation):
... # Assuming observation comes in as a dict with a "mean" key
... mean = observation.get("mean", observation)
... var = torch.eye(self.obs_dim).expand(*mean.shape[:-1], -1, -1)
... return mean, var
>>> wm = TensorDictModule(
... DummyWorldModel(obs_dim),
... in_keys=["action", "observation"],
... out_keys=[("next", "observation", "mean"), ("next", "observation", "var")],
... )
>>> imagined_env = ImaginedEnv(wm, base_env, next_observation_key=("next", "observation"))
>>> # Collect an imagined rollout
>>> rollout = imagined_env.rollout(max_steps=5, policy=RandomPolicy(imagined_env.action_spec))
```