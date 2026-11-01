# MOGymEnv

torchrl.envs.MOGymEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/gym.html#MOGymEnv)

FARAMA MO-Gymnasium environment wrapper.

Examples

```
>>> env = MOGymEnv(env_name="minecart-v0", frame_skip=4)
>>> td = env.rand_step()
>>> print(td)
>>> print(env.available_envs)
```