# JumanjiEnv

torchrl.envs.JumanjiEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/jumanji.html#JumanjiEnv)

Jumanji environment wrapper built with the environment name.

Jumanji offers a vectorized simulation framework based on Jax.
TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
but computational graphs can still be built on top of the simulated trajectories,
allowing for backpropagation through the rollout.

GitHub: [instadeepai/jumanji](https://github.com/instadeepai/jumanji)

Doc: [https://instadeepai.github.io/jumanji/](https://instadeepai.github.io/jumanji/)

Paper: [https://arxiv.org/abs/2306.09884](https://arxiv.org/abs/2306.09884)

Parameters:

- **env_name** (*str*) - the name of the environment to wrap. Must be part of `available_envs`.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - Not yet supported.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
With `jumanji`, this indicates the number of vectorized environments.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**available_envs** - environments available to build

Examples

```
>>> from torchrl.envs import JumanjiEnv
>>> env = JumanjiEnv("Snake-v1")
>>> env.set_seed(0)
>>> td = env.reset()
>>> td["action"] = env.action_spec.rand()
>>> td = env.step(td)
>>> print(td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 next: TensorDict(
 fields={
 action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 state: TensorDict(
 fields={
 action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
 body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
 fruit_position: TensorDict(
 fields={
 col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 head_position: TensorDict(
 fields={
 col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
 length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 state: TensorDict(
 fields={
 action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
 body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
 fruit_position: TensorDict(
 fields={
 col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 head_position: TensorDict(
 fields={
 col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
 length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> print(env.available_envs)
['Game2048-v1',
 'Maze-v0',
 'Cleaner-v0',
 'CVRP-v1',
 'MultiCVRP-v0',
 'Minesweeper-v0',
 'RubiksCube-v0',
 'Knapsack-v1',
 'Sudoku-v0',
 'Snake-v1',
 'TSP-v1',
 'Connector-v2',
 'MMST-v0',
 'GraphColoring-v0',
 'RubiksCube-partly-scrambled-v0',
 'RobotWarehouse-v0',
 'Tetris-v0',
 'BinPack-v2',
 'Sudoku-very-easy-v0',
 'JobShop-v0']
```

To take advante of Jumanji, one usually executes multiple environments at the
same time.

```
>>> from torchrl.envs import JumanjiEnv
>>> env = JumanjiEnv("Snake-v1", batch_size=[10])
>>> env.set_seed(0)
>>> td = env.reset()
>>> td["action"] = env.action_spec.rand()
>>> td = env.step(td)
```

In the following example, we iteratively test different batch sizes
and report the execution time for a short rollout:

Examples

```
>>> from torch.utils.benchmark import Timer
>>> for batch_size in [4, 16, 128]:
... timer = Timer(
... '''
... env.rollout(100)
... ''',
... setup=f'''
... from torchrl.envs import JumanjiEnv
... env = JumanjiEnv('Snake-v1', batch_size=[{batch_size}])
... env.set_seed(0)
... env.rollout(2)
... ''')
... print(batch_size, timer.timeit(number=10))
4 <torch.utils.benchmark.utils.common.Measurement object at 0x1fca91910>
env.rollout(100)
setup: [...]
 Median: 122.40 ms
 2 measurements, 1 runs per measurement, 1 thread
16 <torch.utils.benchmark.utils.common.Measurement object at 0x1ff9baee0>
env.rollout(100)
setup: [...]
 Median: 134.39 ms
 2 measurements, 1 runs per measurement, 1 thread
128 <torch.utils.benchmark.utils.common.Measurement object at 0x1ff9ba7c0>
env.rollout(100)
setup: [...]
 Median: 172.31 ms
 2 measurements, 1 runs per measurement, 1 thread
```