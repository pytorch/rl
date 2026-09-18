# EvaluatorHook

*class*torchrl.trainers.EvaluatorHook(*evaluator: [Evaluator](torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator)*, ***, *every_frames: int*, *policy: str | Callable[[[Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)], [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] = 'loss_module.actor_network'*, *run_at_start: bool = False*, *run_at_end: bool = True*, *wait_at_end: bool = True*, *wait_at_end_timeout: float | None = 60.0*)[[source]](../../_modules/torchrl/trainers/trainers.html#EvaluatorHook)

Schedule asynchronous evaluation from a [`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer).

The hook snapshots the training policy when an evaluation is triggered, polls
completed results after each collected batch, and logs them under the
`evaluation/` namespace. If several evaluation intervals elapse while an
evaluation is running, they are coalesced into one evaluation with the latest
policy weights when the evaluator becomes available.

Parameters:

**evaluator** ([*Evaluator*](torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator)) - Evaluator service used to run rollouts.

Keyword Arguments:

- **every_frames** (*int*) - Number of collected frames between evaluations.
- **policy** (*str**or**Callable**,**optional*) - Dot-separated path resolved from the
trainer, or a callable receiving the trainer and returning an
[`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) or [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase).
Defaults to `"loss_module.actor_network"`.
- **run_at_start** (*bool**,**optional*) - Whether to evaluate the initial policy.
Defaults to `False`.
- **run_at_end** (*bool**,**optional*) - Whether to request a final evaluation with
the latest policy weights. A final evaluation is skipped when the
latest completed evaluation already used the same frame count.
Defaults to `True`.
- **wait_at_end** (*bool**,**optional*) - Whether shutdown waits for pending and final
evaluations so their metrics are logged. When `False`, outstanding
work is handed to `Evaluator.shutdown()` and may be cancelled by
the evaluator backend. Defaults to `True`.
- **wait_at_end_timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**None**,**optional*) - Maximum seconds to wait for
a pending evaluation during shutdown. `None` waits without a time
limit. Defaults to `60.0`.

Examples

```
>>> from torchrl.collectors import Evaluator
>>> from torchrl.trainers import EvaluatorHook
>>> evaluator = Evaluator(make_eval_env, eval_policy, max_steps=1_000) 
>>> EvaluatorHook(evaluator, every_frames=10_000).register(trainer)
```

Note

Checkpoints contain only the next due frame and the last completed
evaluation frame. In-flight evaluator work is intentionally not
serialized and is discarded when resuming from a checkpoint.

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'evaluator_hook'*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#EvaluatorHook.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.