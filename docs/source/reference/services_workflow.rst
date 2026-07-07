.. currentmodule:: torchrl.services

Combining TorchRL Services
==========================

.. _ref_services_workflow:

Training applications commonly need several independently deployed services:
actors request policy inference, write transitions to a replay buffer, and
report metrics to a logger. TorchRL gives these components the same ownership
shape without imposing a single communication protocol on all of them.

This page builds a small pipeline with three owners:

.. list-table:: Example deployment
   :header-rows: 1

   * - Owner
     - Backend
     - Worker-side capability
   * - :class:`~torchrl.modules.inference_server.InferenceServer`
     - thread
     - :class:`~torchrl.modules.inference_server.PolicyClientModule`
   * - :class:`~torchrl.data.TensorDictReplayBuffer`
     - direct or Ray
     - replay-buffer client
   * - :class:`~torchrl.record.loggers.CSVLogger`
     - process
     - logger client

The complete runnable source is
`examples/services/multi_service_training.py
<https://github.com/pytorch/rl/blob/main/examples/services/multi_service_training.py>`_.
It uses only core dependencies with a direct replay buffer; the Ray variant
requires ``pip install ray``.

.. code-block:: bash

    python examples/services/multi_service_training.py
    python examples/services/multi_service_training.py --replay-backend ray

Create owners in the driver
---------------------------

The driver constructs each heavy component exactly once. Backend selection is
local to construction and does not leak into the actor loop:

.. code-block:: python

    from functools import partial

    from torchrl.data import ListStorage, TensorDictReplayBuffer
    from torchrl.modules.inference_server import InferenceServer, ThreadingTransport
    from torchrl.record import CSVLogger

    logger = CSVLogger(
        exp_name="multi-service",
        log_dir="/tmp/torchrl-service-example",
        service_backend="process",
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=partial(ListStorage, max_size=1_000),
        batch_size=32,
        service_backend="ray",
        service_backend_options={"remote_config": {"num_cpus": 1}},
    )

    inference_server = InferenceServer(
        policy,
        ThreadingTransport(),
        max_batch_size=8,
    )
    inference_server.start()

For a dependency-free run, set the replay-buffer backend to ``"direct"`` and
omit its backend options. The direct client's identity semantics mean that
``replay_buffer.client() is replay_buffer``; no proxy is added to the local
hot path.

Distribute clients, not owners
------------------------------

Call :meth:`Service.client` once for each independent producer. Remote clients
are small and picklable and expose only their domain operations. In particular,
a worker holding the Ray replay-buffer and process-logger clients below cannot
stop their owners:

.. code-block:: python

    from torchrl.modules.inference_server import PolicyClientModule

    actor_policy = PolicyClientModule(
        inference_server.client(),
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )
    actor_replay_buffer = replay_buffer.client()
    actor_logger = logger.client()

    assert not hasattr(actor_replay_buffer, "shutdown")
    assert not hasattr(actor_logger, "shutdown")

The inference and logger clients used by this example carry their own reply
routing, so each actor receives a separately created client rather than a
copied client already in use by another actor. A direct service instead uses
identity semantics: its client is the owner because no process boundary needs
capability restriction or proxying.

Keep the actor loop backend-neutral
-----------------------------------

The worker does not need to know where any service runs:

.. code-block:: python

    result = actor_policy(observation_td)
    transition.set("action", result["action"])
    actor_replay_buffer.add(transition)
    actor_logger.log_scalar("actor/reward", reward, step=step)

Remote logger methods have the same completion semantics as direct logger
methods: when ``log_scalar`` returns, the concrete logger method has run, and
service-side errors are raised at that call. Custom ``log_*`` methods likewise
return the concrete method's result. :meth:`~torchrl.record.loggers.Logger.flush`
is still useful to flush buffers maintained by the underlying logging SDK, but
it is not required as a command-completion barrier.

Only the driver controls lifecycle
----------------------------------

Shut owners down after their consumers, with the logger last so teardown
metrics can still be recorded:

.. code-block:: python

    try:
        run_training()
    finally:
        inference_server.shutdown()
        replay_buffer.shutdown()
        logger.shutdown()

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
