# set_gym_backend

torchrl.envs.set_gym_backend(*backend*)[[source]](../../_modules/torchrl/envs/libs/gym.html#set_gym_backend)

Sets the gym-backend to a certain value.

Parameters:

**backend** (*python module**,**string**or**callable returning a module*) - the
gym backend to use. Use a string or callable whenever you wish to
avoid importing gym at loading time.

Examples

```
>>> import gym
>>> import gymnasium
>>> with set_gym_backend("gym"):
... assert gym_backend() == gym
>>> with set_gym_backend(lambda: gym):
... assert gym_backend() == gym
>>> with set_gym_backend(gym):
... assert gym_backend() == gym
>>> with set_gym_backend("gymnasium"):
... assert gym_backend() == gymnasium
>>> with set_gym_backend(lambda: gymnasium):
... assert gym_backend() == gymnasium
>>> with set_gym_backend(gymnasium):
... assert gym_backend() == gymnasium
```

This class can also be used as a function decorator.

Examples

```
>>> @set_gym_backend("gym")
... def fun():
... gym = gym_backend()
... print(gym)
>>> fun()
<module 'gym' from '/path/to/env/site-packages/gym/__init__.py'>
>>> @set_gym_backend("gymnasium")
... def fun():
... gym = gym_backend()
... print(gym)
>>> fun()
<module 'gymnasium' from '/path/to/env/site-packages/gymnasium/__init__.py'>
```