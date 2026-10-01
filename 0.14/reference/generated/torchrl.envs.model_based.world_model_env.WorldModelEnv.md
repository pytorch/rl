# WorldModelEnv

torchrl.envs.model_based.world_model_env.WorldModelEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/model_based/world_model_env.html#WorldModelEnv)

A generic environment wrapper around a [`WorldModel`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel).

Wraps a [`WorldModel`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel) so it can be driven through
the standard [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) API and rolled out with
[`rollout()`](torchrl.envs.EnvBase.html#id2). The world model owns prediction
(encoder, dynamics, reward / done heads, optional decoder); this env owns
the rollout contract (reset, step, done handling, spec validation).

Use this class instead of writing a bespoke rollout loop on the world
model itself. The env semantics -- including how
[`rollout()`](torchrl.envs.EnvBase.html#id2) propagates state via
`step_mdp()` and how it terminates on `done` --
are then shared with every other TorchRL env and stay consistent across
real and imagined rollouts.

The env steps in latent space: it does **not** rerun the world model's
encoder on every step. The caller is expected to seed the latent state on
reset, typically by calling `WorldModel.encode()` on an observation
tensordict and passing the result as the `tensordict` argument to
[`reset()`](torchrl.envs.EnvBase.html#id1) or
[`rollout()`](torchrl.envs.EnvBase.html#id2).

Specs are taken from a reference env so that the imagined env presents
the same action / reward / done specs as the real one. The observation
spec defaults to the latent representation (under `latent_key`); pass
`observation_spec=` to override (e.g. when a decoder is present and the
env should expose decoded observations).

Parameters:

- **world_model** ([*WorldModel*](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel)) - the prediction module that the env drives.
Its [`step_module`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel.step_module) is used as
the underlying `world_model` argument of
`ModelBasedEnvBase`.
- **base_env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - a reference env whose action / reward / done
specs are copied into the imagined env. The reference env is not
stepped -- only its specs are read.

Keyword Arguments:

- **observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - override for the observation
spec. When `None`, the env exposes the latent state under
`latent_key` with shape inferred from `base_env`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - batch size for the env. Defaults
to `base_env.batch_size`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device for the env. Defaults to
`base_env.device`.
- **latent_key** (*NestedKey**,**optional*) - the key under which the latent
state is stored. Defaults to `"latent"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs.model_based import WorldModelEnv
>>> from torchrl.modules import WorldModel
>>> base_env = GymEnv("Pendulum-v1")
>>> obs_dim = base_env.observation_spec["observation"].shape[-1]
>>> action_dim = base_env.action_spec.shape[-1]
>>> latent_dim = 4
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
>>> wm_env = WorldModelEnv(world_model, base_env=base_env, batch_size=[3])
>>> # Seed the env with a starting latent and roll it out.
>>> obs_td = TensorDict(
... {"observation": torch.randn(3, obs_dim)}, batch_size=[3]
... )
>>> start_td = world_model.encode(obs_td)
>>> rollout = wm_env.rollout(max_steps=5, tensordict=start_td)
>>> rollout.shape
torch.Size([3, 5])
```