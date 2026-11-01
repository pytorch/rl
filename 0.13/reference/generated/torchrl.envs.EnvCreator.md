# EnvCreator

*class*torchrl.envs.EnvCreator(*create_env_fn: Callable[[...], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]*, *create_env_kwargs: dict | None = None*, *share_memory: bool = True*, ***kwargs*)[[source]](../../_modules/torchrl/envs/env_creator.html#EnvCreator)

Environment creator class.

EnvCreator is a generic environment creator class that can substitute
lambda functions when creating environments in multiprocessing contexts.
If the environment created on a subprocess must share information with the
main process (e.g. for the VecNorm transform), EnvCreator will pass the
pointers to the tensordicts in shared memory to each process such that
all of them are synchronised.

Parameters:

- **create_env_fn** (*callable*) - a callable that returns an EnvBase
instance.
- **create_env_kwargs** (*dict**,**optional*) - the kwargs of the env creator.
- **share_memory** (*bool**,**optional*) - if False, the resulting tensordict
from the environment won't be placed in shared memory.
- ****kwargs** - additional keyword arguments to be passed to the environment
during construction.

Examples

```
>>> # We create the same environment on 2 processes using VecNorm
>>> # and check that the discounted count of observations matches on
>>> # both workers, even if one has not executed any step
>>> import time
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.envs.transforms import VecNorm, TransformedEnv
>>> from torchrl.envs import EnvCreator
>>> from torch import multiprocessing as mp
>>> env_fn = lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm())
>>> env_creator = EnvCreator(env_fn)
>>>
>>> def test_env1(env_creator):
... env = env_creator()
... tensordict = env.reset()
... for _ in range(10):
... env.rand_step(tensordict)
... if tensordict.get(("next", "done")):
... tensordict = env.reset(tensordict)
... print("env 1: ", env.transform._td.get(("next", "observation_count")))
>>>
>>> def test_env2(env_creator):
... env = env_creator()
... time.sleep(5)
... print("env 2: ", env.transform._td.get(("next", "observation_count")))
>>>
>>> if __name__ == "__main__":
... ps = []
... p1 = mp.Process(target=test_env1, args=(env_creator,))
... p1.start()
... ps.append(p1)
... p2 = mp.Process(target=test_env2, args=(env_creator,))
... p2.start()
... ps.append(p1)
... for p in ps:
... p.join()
env 1: tensor([11.9934])
env 2: tensor([11.9934])
```

make_variant(***kwargs*) → EnvCreator[[source]](../../_modules/torchrl/envs/env_creator.html#EnvCreator.make_variant)

Creates a variant of the EnvCreator, pointing to the same underlying metadata but with different keyword arguments during construction.

This can be useful with transforms that share a state, like `TrajCounter`.

Examples

```
>>> from torchrl.envs import GymEnv
>>> env_creator_pendulum = EnvCreator(GymEnv, env_name="Pendulum-v1")
>>> env_creator_cartpole = env_creator_pendulum.make_variant(env_name="CartPole-v1")
```