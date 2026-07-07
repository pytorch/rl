.. currentmodule:: torchrl.services

.. _ref_services_workflow:

Combining TorchRL Services
==========================

Training applications commonly need several independently deployed services:
actors request policy inference, write transitions to a replay buffer, and
report metrics to a logger. TorchRL gives these components the same ownership
shape without imposing a single communication protocol on all of them.

The examples run one ordinary environment/replay/training loop under three
deployment profiles. Each entry point is intentionally short and delegates
the same TensorDict training logic to ``multi_service_utils.py``:

.. list-table:: Example profiles
   :header-rows: 1

   * - Entry point
     - Inference
     - Logger
     - Replay buffer
     - Training loop
   * - ``multi_service_single_process.py``
     - background thread
     - direct
     - direct
     - driver
   * - ``multi_service_multiprocess.py``
     - spawned process
     - spawned process
     - direct
     - driver
   * - ``multi_service_ray.py``
     - Ray transport to a driver-owned server
     - Ray actor
     - Ray actor
     - driver

The runnable sources are:

- `single-process example
  <https://github.com/pytorch/rl/blob/main/examples/services/multi_service_single_process.py>`_;
- `multiprocess example
  <https://github.com/pytorch/rl/blob/main/examples/services/multi_service_multiprocess.py>`_;
- `Ray example
  <https://github.com/pytorch/rl/blob/main/examples/services/multi_service_ray.py>`_;
- `shared training helper
  <https://github.com/pytorch/rl/blob/main/examples/services/multi_service_utils.py>`_.

All three require Gymnasium; the third additionally requires Ray.

.. code-block:: bash

    pip install gymnasium
    python examples/services/multi_service_single_process.py
    python examples/services/multi_service_multiprocess.py
    pip install ray
    python examples/services/multi_service_ray.py

Create owners in the driver
---------------------------

The driver constructs each heavy component exactly once. The three entry
points differ only in the profile passed to the helper:

.. code-block:: python

    from multi_service_utils import run_training

    run_training(service_backend="direct")
    run_training(service_backend="process")
    run_training(service_backend="ray")

The helper resolves the profile into the backends currently supported by each
domain. The process profile uses process owners for inference and logging but
keeps the replay buffer in the driver because replay buffers do not yet expose
a process service backend. The direct client's identity semantics mean that
``replay_buffer.client() is replay_buffer``; no proxy is added to that hot
path. The Ray profile uses restricted Ray clients for both logging and replay.

Use clients in the loop, not owners
-----------------------------------

Remote clients expose the same domain operations as their direct counterparts,
without lifecycle methods. The helper extracts them once before entering the
loop; :class:`~torchrl.modules.inference_server.PolicyClientModule` performs
the same extraction automatically for the inference owner:

.. code-block:: python

    from torchrl.modules.inference_server import PolicyClientModule

    policy = PolicyClientModule(
        inference_owner,
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )
    replay_buffer = replay_owner.client()
    logger = logger_owner.client()

A direct service uses identity semantics: its client is the owner because no
process boundary needs capability restriction or proxying. Process and Ray
profiles return lightweight clients that transparently cross their respective
boundaries.

Keep the training loop backend-neutral
--------------------------------------

The core loop does not need to know where any service runs. Policy inference,
replay writes and samples, and logging retain their normal TorchRL interfaces:

.. code-block:: python

    td = env.reset()
    for step in range(steps):
        td = policy(td)
        step_td = env.step(td)
        replay_buffer.add(step_td)
        td = env.step_mdp(step_td)

        sample = replay_buffer.sample()
        optimizer.zero_grad()
        loss = loss_fn(sample)
        loss.sum(reduce=True).backward()
        optimizer.step()

        logger.log_scalar(
            "train/loss", float(loss["loss"].detach()), step=step
        )

``loss_fn`` returns a TensorDict of loss terms, as TorchRL objective modules
do. ``reduce=True`` reduces all of those terms to one scalar tensor before
backpropagation. Nothing in this loop branches on ``service_backend``; the
owners created before the loop determine which calls are local and which cross
a process or Ray boundary.

Remote logger methods have the same completion semantics as direct logger
methods: when ``log_scalar`` returns, the concrete logger method has run, and
service-side errors are raised at that call. Custom ``log_*`` methods likewise
return the concrete method's result. :meth:`~torchrl.record.loggers.Logger.flush`
is still useful to flush buffers maintained by the underlying logging SDK, but
it is not required as a command-completion barrier.

Only the driver controls lifecycle
----------------------------------

The helper uses an :class:`contextlib.ExitStack` to shut owners down after
their consumers, with the logger last so teardown metrics can still be
recorded:

.. code-block:: python

    from contextlib import ExitStack

    with ExitStack() as owners:
        logger = make_logger()
        owners.callback(logger.shutdown)
        replay_buffer = make_replay_buffer()
        owners.callback(replay_buffer.shutdown)
        inference_server = make_inference_server()
        owners.callback(inference_server.shutdown)
        run_training_loop()

Shutdown is idempotent. Explicit teardown also releases process queues, feeder
threads, and Ray actors deterministically instead of relying on garbage
collection.

Add evaluation video without service plumbing
---------------------------------------------

:class:`~torchrl.record.recorder.VideoRecorder` accepts a logger owner and
obtains its restricted client itself. A vector environment is recorded as one
synchronized grid, using the root ``"pixels"`` key by default:

.. code-block:: python

    from torchrl.envs import TransformedEnv
    from torchrl.record import VideoRecorder

    eval_env = TransformedEnv(
        parallel_eval_env,
        VideoRecorder(logger, tag="eval/video", fps=30),
    )

No logger-client extraction, recorder lookup, input-key override, or grid
configuration is needed. Calling ``env.transform.dump(step=step)`` writes the
video; :class:`~torchrl.collectors.Evaluator` performs the same dump through
its collector RPC path when evaluation runs in another process.

Discovery is optional
---------------------

The example passes clients explicitly because their destinations are known at
startup. :func:`get_services` is an orthogonal discovery mechanism for clients
that must be found dynamically by Ray workers. Registering an existing owner
stores its restricted client without transferring ownership; resetting the
registry removes discovery entries but does not shut that owner down. See
:ref:`ref_services` for registry usage and namespace isolation.
