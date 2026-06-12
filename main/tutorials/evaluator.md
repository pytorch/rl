Note

Go to the end
to download the full example code.

# Using the Evaluator

**Author**: [Vincent Moens](https://github.com/vmoens)

 What you will learn

- How to run synchronous and asynchronous evaluations during training
- How to pass updated weights to the evaluator
- How to use the `on_result` callback for logging
- How to run evaluation in a separate process

 Prerequisites

- [TorchRL](https://github.com/pytorch/rl) and
[gymnasium](https://gymnasium.farama.org) installed
- Familiarity with [`EnvBase`](../reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) and
[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector)

```

```

In RL training loops, evaluation is often done inline: you stop training,
run a few rollouts, log the metrics, then resume. This blocks the
training loop while rollouts are collected. For environments with
expensive step functions (robotics simulators, LLM generation, etc.),
this can waste significant GPU time.

The [`Evaluator`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator) decouples evaluation from
training by running rollouts in the background and letting you poll for
metrics or react to results via a callback.

In this tutorial we will cover:

1. Synchronous evaluation -- blocking calls
2. Asynchronous evaluation -- fire-and-poll
3. Weight updates -- passing trained weights
4. Process-based evaluation -- out-of-process
5. Logging with callbacks -- `on_result`

```
from functools import partial

import torch
from tensordict import from_module
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Evaluator, RandomPolicy
from torchrl.envs import GymEnv
```

## Synchronous evaluation

The simplest way to use the Evaluator is to call
[`evaluate()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.evaluate), which blocks until
the rollout completes and returns a metrics dict.

We start by creating an environment factory and a random policy.
The Evaluator accepts either a live environment or a callable that
creates one -- the callable form is preferred because it lets the
evaluator recreate the environment if needed.

```
env_maker = partial(GymEnv, "Pendulum-v1")
policy = RandomPolicy(env_maker().action_spec)
evaluator = Evaluator(env_maker, policy, num_trajectories=1)
```

Now we can run a blocking evaluation. The returned dict contains
prefixed metrics: reward, episode length, number of episodes, and
frames-per-second.

```
result = evaluator.evaluate()
print("First eval:", result)
```

```
First eval: {'eval/reward': -1565.214599609375, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 687.0636748456759, 'eval/step': 0}
```

Each subsequent call increments the internal step counter:

```
result = evaluator.evaluate()
print("Second eval:", result)
```

```
Second eval: {'eval/reward': -856.1010131835938, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 691.5509353007501, 'eval/step': 1}
```

## Asynchronous evaluation

For non-blocking evaluation, use [`trigger_eval()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.trigger_eval)
to start a rollout in the background, then [`poll()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.poll)
or [`wait()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.wait) to retrieve the result.

```
evaluator.trigger_eval()

# poll() is non-blocking: returns None if the result isn't ready yet
result = evaluator.poll()
print("poll() returned:", result)
```

```
poll() returned: None
```

To wait for the result, pass a timeout to `poll()` or use `wait()`:

```
result = evaluator.poll(timeout=30)
print("poll(timeout=30) returned:", result)
```

```
poll(timeout=30) returned: {'eval/reward': -996.3477783203125, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 706.6117167239654, 'eval/step': 2}
```

By default, calling `trigger_eval()` while a previous evaluation is
still pending raises an error. This prevents silently piling up stale
requests:

```
evaluator.trigger_eval()
try:
 evaluator.trigger_eval()
except RuntimeError as e:
 print(f"Errored with: {e}")

# Clean up
evaluator.wait(timeout=30)
evaluator.shutdown()
```

If you prefer to enqueue evaluations, pass `busy_policy="queue"`
when creating the evaluator.

## Weight updates

In a real training loop, you want to evaluate the *latest* trained
weights, not the initial ones. The [`evaluate()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.evaluate)
and [`trigger_eval()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.trigger_eval) methods accept a
`weights` argument -- either an `nn.Module` or a `TensorDictBase`.

Let's create a simple MLP policy and an evaluator for it:

```
env = env_maker()
net = nn.Sequential(
 nn.Linear(env.observation_spec["observation"].shape[-1], 64),
 nn.Tanh(),
 nn.Linear(64, env.action_spec.shape[-1]),
)
real_policy = TensorDictModule(net, in_keys=["observation"], out_keys=["action"])

evaluator_w = Evaluator(env_maker, real_policy, num_trajectories=1)
```

Evaluate with the initial (random) weights:

```
print("Before weight update:", evaluator_w.evaluate())
```

```
Before weight update: {'eval/reward': -1032.2110595703125, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 500.0935374952034, 'eval/step': 0}
```

Simulate a "training step" by perturbing the weights:

```
with torch.no_grad():
 for p in net.parameters():
 p.add_(torch.randn_like(p) * 0.1)
```

Now evaluate with the updated weights. You can pass the module
directly -- the evaluator extracts and transfers the weights
automatically:

```
print("After weight update:", evaluator_w.evaluate(weights=real_policy))
```

```
After weight update: {'eval/reward': -1904.338134765625, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 497.22991845047716, 'eval/step': 1}
```

You can also pass a `TensorDictBase` of weights, which is useful
when you already have detached weight snapshots:

```
real_weights = from_module(real_policy)
print("With TensorDict weights:", evaluator_w.evaluate(weights=real_weights))
evaluator_w.shutdown()
```

```
With TensorDict weights: {'eval/reward': -1658.74951171875, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 490.7984962945619, 'eval/step': 2}
```

## Process-based evaluation

For full isolation (e.g. to place evaluation on a dedicated GPU or to
avoid GIL contention), use `backend="process"`. This runs the
environment and policy inside a child process via
[`MultiSyncCollector`](../reference/generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector).

The process backend requires callable factories for both the
environment and the policy:

```
env_maker = partial(GymEnv, "Pendulum-v1")
action_spec = env_maker().action_spec
policy_factory = partial(RandomPolicy, action_spec)

evaluator_proc = Evaluator(
 env_maker,
 policy_factory=policy_factory,
 num_trajectories=1,
 backend="process",
)

result = evaluator_proc.evaluate()
print("Process backend:", result)
evaluator_proc.shutdown()
```

```
Process backend: {'eval/reward': -1612.5433349609375, 'eval/reward_std': 0.0, 'eval/num_episodes': 1, 'eval/episode_length': 200.0, 'eval/fps': 687.8832586717347, 'eval/step': 0}
```

## Logging with callbacks

Rather than manually logging after each `poll()` or `wait()`, you
can pass an `on_result` callback to the evaluator. It receives a flat
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) with the same prefixed metric names.

Here we use TorchRL's [`CSVLogger`](../reference/generated/torchrl.record.loggers.csv.CSVLogger.html#torchrl.record.loggers.csv.CSVLogger) to
automatically log every evaluation result to a CSV file:

```
import tempfile

from torchrl.record.loggers.csv import CSVLogger

log_dir = tempfile.mkdtemp()
logger = CSVLogger(exp_name="eval_demo", log_dir=log_dir)

evaluator_log = Evaluator(
 env_maker,
 real_policy,
 num_trajectories=1,
 on_result=lambda result: logger.log_metrics(
 {k: v.item() for k, v in result.items() if k != "eval/step"},
 step=result["eval/step"].item(),
 ),
)
```

Run a few evals. Each one automatically logs to CSV via the callback:

```
for _ in range(3):
 evaluator_log.evaluate(weights=real_policy)

evaluator_log.shutdown()
```

Let's verify what was logged:

```
from pathlib import Path

csv_path = next(Path(log_dir).rglob("*.csv"))
print(f"Logged to: {csv_path}")
print(csv_path.read_text())
```

```
Logged to: /tmp/tmpvbv_j17y/eval_demo/scalars/eval/reward.csv
0,-1736.2789306640625
1,-1511.2886962890625
2,-1389.80029296875
```

The `on_result` callback works with both synchronous and asynchronous
evaluation. For async usage, the callback runs on the evaluator's
background thread -- if your callback writes to a shared logger, handle
any required locking inside the callback.

## Summary

The [`Evaluator`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator) provides a single,
composable entry-point for evaluation:

- **Synchronous**: [`evaluate()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.evaluate) for
blocking rollouts.
- **Asynchronous**: [`trigger_eval()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.trigger_eval)
+ [`poll()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.poll) /
[`wait()`](../reference/generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.wait) for background rollouts.
- **Weight sync**: pass `weights` (module or tensordict) to evaluate
the latest trained parameters.
- **Process isolation**: `backend="process"` for dedicated-device eval.
- **Callbacks**: `on_result` for automatic logging or checkpointing.

### Useful next resources

- [Evaluator API reference](../reference/collectors_eval.html#collectors-eval) -- full parameter docs.
- [Collector trajectory tutorial](collector_trajectory_assembly.html#collector-trajectory-assembly) --
deep dive into how collectors assemble data.
- [TorchRL documentation](https://pytorch.org/rl/)

**Total running time of the script:** (0 minutes 7.034 seconds)

[`Download Jupyter notebook: evaluator.ipynb`](../_downloads/0b7694ad40d411d9b5498dc6c93ed596/evaluator.ipynb)

[`Download Python source code: evaluator.py`](../_downloads/a2327df13c566c4551caecc730bd4065/evaluator.py)

[`Download zipped: evaluator.zip`](../_downloads/442d647775694bcab842ec117ca0cc74/evaluator.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)