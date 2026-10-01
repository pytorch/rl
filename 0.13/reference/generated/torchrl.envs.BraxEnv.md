# BraxEnv

torchrl.envs.BraxEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/brax.html#BraxEnv)

Google Brax environment wrapper built with the environment name.

Brax offers a vectorized and differentiable simulation framework based on Jax.
TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
but computational graphs can still be built on top of the simulated trajectories,
allowing for backpropagation through the rollout.

GitHub: [google/brax](https://github.com/google/brax)

Paper: [https://arxiv.org/abs/2106.13281](https://arxiv.org/abs/2106.13281)

Parameters:

- **env_name** (*str*) - the environment name of the env to wrap. Must be part of
`available_envs`.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.
- **cache_clear_frequency** (*int**,**optional*) - automatically clear JAX's internal
cache every N steps to prevent memory leaks when using `requires_grad=True`.
Defaults to False (deactivates automatic cache clearing).

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - Not yet supported.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
In `brax`, this controls the number of environments simulated in
parallel via JAX's `vmap` on a single device (GPU/TPU). Brax leverages
MuJoCo XLA (MJX) for hardware-accelerated batched simulation, enabling
thousands of environments to run in parallel within a single process.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.
- **num_workers** (*int**,**optional*) - if greater than 1, a lazy [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv)
will be returned instead, with each worker instantiating its own
`BraxEnv` instance. Defaults to `None`.

Note

There are two orthogonal ways to scale environment throughput:

- **batch_size**: Uses Brax's native JAX-based vectorization (`vmap`) to run
multiple environments in parallel on a single GPU/TPU. This is highly efficient
for moderate batch sizes where the MJX solver has not yet saturated.
- **num_workers**: Uses TorchRL's [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) to spawn
multiple Python processes, each running its own `BraxEnv`.

These can be combined: `BraxEnv("ant", batch_size=[128], num_workers=4)` creates
4 worker processes, each running 128 vectorized environments, for a total of 512
parallel environments. This hybrid approach can be beneficial when the MJX solver
saturates on a single device, or when distributing across multiple GPUs/CPUs.

Variables:

**available_envs** - environments available to build

Examples

```
>>> from torchrl.envs import BraxEnv
>>> import torch
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> env = BraxEnv("ant", device=device)
>>> env.set_seed(0)
>>> td = env.reset()
>>> td["action"] = env.action_spec.rand()
>>> td = env.step(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(torch.Size([8]), dtype=torch.float32),
 done: Tensor(torch.Size([1]), dtype=torch.bool),
 next: TensorDict(
 fields={
 observation: Tensor(torch.Size([87]), dtype=torch.float32)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 observation: Tensor(torch.Size([87]), dtype=torch.float32),
 reward: Tensor(torch.Size([1]), dtype=torch.float32),
 state: TensorDict(...)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> print(env.available_envs)
['acrobot', 'ant', 'fast', 'fetch', ...]
```

# Example: create a parallel environment with 4 workers. This returns a lazy
# ParallelEnv; each worker will instantiate a BraxEnv with num_workers=1.
>>> from torchrl.envs import BraxEnv
>>> par_env = BraxEnv("ant", batch_size=[8], num_workers=4, device="cpu")
>>> # par_env is a ParallelEnv; start interacting as usual
>>> par_env.set_seed(0)
>>> td = par_env.reset()
>>> print(td.shape)
torch.Size([4, 8])
>>> td["action"] = par_env.action_spec.rand()
>>> td = par_env.step(td)

To take advante of Brax, one usually executes multiple environments at the
same time. In the following example, we iteratively test different batch sizes
and report the execution time for a short rollout:

Examples

```
>>> import torch
>>> from torch.utils.benchmark import Timer
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> for batch_size in [4, 16, 128]:
... timer = Timer('''
... env.rollout(100)
... ''',
... setup=f'''
... from torchrl.envs import BraxEnv
... env = BraxEnv("ant", batch_size=[{batch_size}], device="{device}")
... env.set_seed(0)
... env.rollout(2)
... ''')
... print(batch_size, timer.timeit(10))
4
env.rollout(100)
setup: [...]
310.00 ms
1 measurement, 10 runs , 1 thread
```

16
env.rollout(100)
setup: [...]
268.46 ms
1 measurement, 10 runs , 1 thread

128
env.rollout(100)
setup: [...]
433.80 ms
1 measurement, 10 runs , 1 thread

One can backpropagate through the rollout and optimize the policy directly:

```
>>> from torchrl.envs import BraxEnv
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> import torch
>>>
>>> env = BraxEnv("ant", batch_size=[10], requires_grad=True, cache_clear_frequency=100)
>>> env.set_seed(0)
>>> torch.manual_seed(0)
>>> policy = TensorDictModule(nn.Linear(27, 8), in_keys=["observation"], out_keys=["action"])
>>>
>>> td = env.rollout(10, policy)
>>>
>>> td["next", "reward"].mean().backward(retain_graph=True)
>>> print(policy.module.weight.grad.norm())
tensor(213.8605)
```