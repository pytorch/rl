# gym_backend

torchrl.envs.gym_backend(*submodule=None*)[[source]](../../_modules/torchrl/envs/libs/gym.html#gym_backend)

Returns the gym backend, or a submodule of it.

Parameters:

**submodule** (*str*) - the submodule to import. If `None`, the backend
itself is returned.

Examples

```
>>> import mo_gymnasium
>>> with set_gym_backend("gym"):
... wrappers = gym_backend('wrappers')
... print(wrappers)
>>> with set_gym_backend("gymnasium"):
... wrappers = gym_backend('wrappers')
... print(wrappers)
```