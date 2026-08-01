# MOGymWrapper

torchrl.envs.MOGymWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/gym.html#MOGymWrapper)

FARAMA MO-Gymnasium environment wrapper.

Examples

```
>>> import mo_gymnasium as mo_gym
>>> env = MOGymWrapper(mo_gym.make('minecart-v0'), frame_skip=4)
>>> td = env.rand_step()
>>> print(td)
```