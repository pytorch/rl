# IsaacGymEnv

torchrl.envs.IsaacGymEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/isaacgym.html#IsaacGymEnv)

A TorchRL Env interface for IsaacGym environments.

See `IsaacGymWrapper` for more information.

Examples

```
>>> env = IsaacGymEnv(task="Ant", num_envs=2000, device="cuda:0")
>>> rollout = env.rollout(3)
>>> assert env.batch_size == (2000,)
```