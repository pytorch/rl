.. currentmodule:: torchrl.collectors

.. _collectors_eval:

Evaluation
==========

The :class:`Evaluator` class provides a unified interface for running evaluation
rollouts during RL training, either **synchronously** (blocking) or
**asynchronously** (in a background thread or Ray actor).

Why use an Evaluator?
---------------------

In most RL training loops, evaluation is done inline and **blocks** the training
loop while rollouts are collected.  For environments with expensive step
functions (robotics simulators, LLM generation, etc.) this can waste
significant training time.  The :class:`Evaluator` decouples evaluation from
training by running rollouts in the background and letting you poll for metrics
or react to results via a callback.

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
        on_result=lambda result: my_logger.log_metrics(
            {k: v.item() for k, v in result.items() if k != "eval/step"},
            step=result["eval/step"].item(),
        ),
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
            print(result)  # {"eval/reward": ..., "eval/episode_length": ..., "eval/fps": ...}

    evaluator.shutdown()

Synchronous usage
-----------------

If you prefer blocking evaluation (e.g. for final evaluation or simple
scripts), use :meth:`~Evaluator.evaluate`:

.. code-block:: python

    metrics = evaluator.evaluate(weights=train_policy, step=step)
    # metrics == {"eval/reward": -123.4, "eval/episode_length": 1000, "eval/fps": ...}

Asynchronous usage
------------------

For non-blocking evaluation during training:

.. code-block:: python

    # Start eval in the background
    evaluator.trigger_eval(weights=train_policy, step=step)

    # ... continue training ...

    # Check if results are ready (non-blocking)
    result = evaluator.poll()          # returns None if still running

    # Or block until done
    result = evaluator.wait(timeout=60)

By default, :meth:`~Evaluator.trigger_eval` raises if a previous evaluation is
still pending. Set ``busy_policy="queue"`` to enqueue later requests instead.

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
    )

The ``device`` parameter controls where policy weights are moved before
each rollout.  When passing weights from the training policy (which may
live on a different device), the Evaluator automatically moves them to
the eval device.

Overlap policy (backpressure)
-----------------------------

Calling :meth:`~Evaluator.trigger_eval` while a previous evaluation is
still pending raises immediately by default (``busy_policy="error"``).
This keeps training loops from silently piling up stale evaluation requests.

If you prefer to enqueue evaluations, pass ``busy_policy="queue"``.
Queued requests are processed in order as earlier evaluations finish.

Result callbacks
----------------

Pass ``on_result`` to react to completed evaluations without manual
``poll()`` bookkeeping:

.. code-block:: python

    def on_eval(result):
        metrics = {k: v.item() for k, v in result.items() if k != "eval/step"}
        if metrics["eval/reward"] > best_reward:
            save_checkpoint(step=result["eval/step"].item())

    evaluator = Evaluator(env, policy, max_steps=1000, on_result=on_eval)

For asynchronous evaluations, ``on_result`` runs on the evaluator's
background coordination thread. If your callback talks to a logger that
is also used by the training loop, handle any required locking inside the
callback.

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

Or use ``on_result`` to consume the prefixed evaluation metrics as a flat
tensordict:

.. code-block:: python

    def on_eval(result):
        if result["eval/reward"].item() > best_reward:
            save_checkpoint(result["eval/step"].item())

    evaluator = Evaluator(env, policy, max_steps=1000, on_result=on_eval)

Comparing policies
------------------

To compare policies, environments (for instance the same task in two
simulators) or perturbations, evaluate each combination with
``episode_sampling="per_env"``, ``return_episodes=True`` and a fixed set of
seeds, then reduce the per-episode returns with :func:`bootstrap_estimate` and
:func:`paired_comparison`.

``episode_sampling="per_env"`` makes every rollout contribute exactly one
episode per sub-environment. The default (``"first_done"``) keeps the first
episodes to finish, which over-samples sub-environments with short episodes
when more episodes than sub-environments are requested. Passing the same
``seed`` to :meth:`~Evaluator.evaluate` starts every policy from the same
states, so per-episode results are paired across policies.

.. code-block:: python

    import torch
    from torchrl.collectors import Evaluator, bootstrap_estimate, paired_comparison

    policies = {"ppo": ppo_policy, "sac": sac_policy}
    envs = {"gymnasium": make_gym_env, "mujoco": make_mujoco_env}
    seeds = range(5)

    returns = {}
    for env_name, make_env in envs.items():
        for policy_name, policy in policies.items():
            evaluator = Evaluator(
                make_env(),
                policy,
                num_trajectories=10,
                max_steps=1000,
                episode_sampling="per_env",
                return_episodes=True,
            )
            returns[env_name, policy_name] = torch.cat(
                [
                    evaluator.evaluate(seed=seed)["eval/episodes"]["episode_reward"]
                    for seed in seeds
                ]
            )

    for env_name in envs:
        summary = bootstrap_estimate(returns[env_name, "ppo"], statistic="iqm")
        versus = paired_comparison(returns[env_name, "ppo"], returns[env_name, "sac"])
        logger.log_metrics(
            {
                f"{env_name}/ppo/iqm": summary["estimate"],
                f"{env_name}/ppo_minus_sac": versus["estimate"],
                f"{env_name}/ppo_beats_sac": versus["probability_of_improvement"],
            }
        )

Perturbations follow the same pattern: make them part of the environment
factory (for instance ``lambda: make_env().append_transform(FrameSkipTransform(2))``).

API Reference
-------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Evaluator
    bootstrap_estimate
    paired_comparison
