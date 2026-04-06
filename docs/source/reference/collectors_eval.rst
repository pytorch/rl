.. currentmodule:: torchrl.collectors

.. _collectors_eval:

Evaluation
==========

The :class:`Evaluator` class provides a unified interface for running evaluation
rollouts during RL training, either **synchronously** (blocking) or
**asynchronously** (fire-and-forget in a background thread or Ray actor).

Why use an Evaluator?
---------------------

In most RL training loops, evaluation is done inline and **blocks** the training
loop while rollouts are collected.  For environments with expensive step
functions (robotics simulators, LLM generation, etc.) this can waste
significant training time.  The :class:`Evaluator` decouples evaluation from
training by running rollouts in the background, logging results automatically,
and letting you poll for metrics at your convenience.

Quick example
-------------

.. code-block:: python

    from torchrl.collectors import Evaluator
    from torchrl.envs import GymEnv
    from tensordict.nn import TensorDictModule
    import torch.nn as nn

    def make_eval_env():
        return GymEnv("HalfCheetah-v4")

    eval_policy = TensorDictModule(
        nn.Linear(17, 6), in_keys=["observation"], out_keys=["action"],
    )

    evaluator = Evaluator(
        make_eval_env,
        eval_policy,
        max_steps=1000,
        logger=my_logger,       # auto-logs eval/reward, eval/episode_length
    )

    # --- Inside training loop ---
    for data in collector:
        train(data)

        if should_eval:
            # Non-blocking: kick off eval and move on
            evaluator.trigger_eval(weights=train_policy, step=collected_frames)

        # Optionally check for results
        result = evaluator.poll()
        if result is not None:
            print(result)  # {"eval/reward": ..., "eval/episode_length": ...}

    evaluator.shutdown()

Synchronous usage
-----------------

If you prefer blocking evaluation (e.g. for final evaluation or simple
scripts), use :meth:`~Evaluator.evaluate`:

.. code-block:: python

    metrics = evaluator.evaluate(weights=train_policy, step=step)
    # metrics == {"eval/reward": -123.4, "eval/episode_length": 1000}

Asynchronous usage
------------------

For non-blocking evaluation during training:

.. code-block:: python

    # Fire-and-forget: starts eval in a background thread
    evaluator.trigger_eval(weights=train_policy, step=step)

    # ... continue training ...

    # Check if results are ready (non-blocking)
    result = evaluator.poll()          # returns None if still running

    # Or block until done
    result = evaluator.wait(timeout=60)

**Fire-and-forget semantics**: calling :meth:`~Evaluator.trigger_eval` while a
previous evaluation is still running discards the in-progress result and starts
the new one.

Device placement and compilation
--------------------------------

For best performance, place the eval policy on a **dedicated device** and
optionally ``torch.compile`` both the env and policy independently of the
training pipeline:

.. code-block:: python

    import torch

    eval_device = torch.device("cuda:1")  # training on cuda:0

    eval_policy = make_policy().to(eval_device)
    eval_env = make_env(device=eval_device)

    # Optional: compile for extra speed
    eval_policy = torch.compile(eval_policy)

    evaluator = Evaluator(
        eval_env,
        eval_policy,
        max_steps=1000,
        device=eval_device,
        logger=logger,
    )

The ``device`` parameter controls where policy weights are moved before
each rollout.  When passing weights from the training policy (which may
live on a different device), the Evaluator automatically moves them to
the eval device.

Overlap policy (backpressure)
-----------------------------

Calling :meth:`~Evaluator.trigger_eval` while a previous evaluation is
still running **drops** the in-progress result (fire-and-forget).  The new
evaluation starts as soon as the background thread finishes the current
``env.rollout()`` call.  There is no queue, no coalescing, and no error.
Only the most recently triggered evaluation produces a result.

This design means you can safely call ``trigger_eval`` on every training
iteration without worrying about a backlog of pending evaluations.

Thread safety of logging
------------------------

All logger writes (scalar metrics and video encoding) happen on the
**caller thread** inside :meth:`~Evaluator.poll`, :meth:`~Evaluator.wait`,
or :meth:`~Evaluator.evaluate`.  The background thread only computes plain
metrics; it never touches the logger.  If you share a logger between
training and evaluation, pass the same ``logger_lock`` to serialise writes.

Backends
--------

The :class:`Evaluator` supports two backends selected via the ``backend``
parameter:

**Thread backend** (default, ``backend="thread"``):

- Runs ``env.rollout()`` in a daemon thread within the same process.
- No extra dependencies required.
- Best for most single-node training setups.

**Ray backend** (``backend="ray"``):

- Wraps :class:`~torchrl.collectors.distributed.RayEvalWorker` under the same
  API.
- Runs evaluation in a separate Ray actor process with its own GPU.
- Required when the eval environment needs process-level initialisation
  (e.g. Isaac Lab's ``AppLauncher``).

.. code-block:: python

    evaluator = Evaluator(
        make_eval_env,
        policy_factory=make_eval_policy,
        max_steps=1000,
        backend="ray",
        init_fn=my_process_init,
        num_gpus=1,
    )

Custom metrics and callbacks
----------------------------

Pass a ``metrics_fn`` to extract custom metrics from rollout data:

.. code-block:: python

    def my_metrics(rollout_td):
        return {
            "success_rate": (rollout_td["next", "success"].any(-1).float().mean().item()),
        }

    evaluator = Evaluator(env, policy, max_steps=1000, metrics_fn=my_metrics)

Or use a ``callback`` for side effects (e.g. saving checkpoints on good evals):

.. code-block:: python

    def on_eval(metrics, step):
        if metrics["eval/reward"] > best_reward:
            save_checkpoint(step)

    evaluator = Evaluator(env, policy, max_steps=1000, callback=on_eval)

API Reference
-------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Evaluator
