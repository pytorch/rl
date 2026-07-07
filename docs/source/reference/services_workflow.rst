.. currentmodule:: torchrl.services

.. _ref_services_workflow:

Designing Training Applications with Services
=============================================

Distributed reinforcement learning combines components with very different
resource and communication requirements. Actors need low-latency policy
inference, replay buffers coordinate concurrent producers and consumers, and
loggers own stateful SDKs, files, credentials, and artifact uploads. These
components often need to live in different threads, processes, or cluster
actors while remaining easy to use from the training loop.

Making an object remote is not sufficient. A useful distributed API must also
answer:

* Who constructs and owns the expensive resource?
* What can safely be copied or pickled into a worker?
* Which operations are available to workers, and which remain driver-only?
* When is an operation complete, and where are failures reported?
* Who shuts down child processes, actors, queues, and background threads?

TorchRL services answer these questions with a common ownership model while
preserving the domain API of each component.

Owners and clients
------------------

A :class:`Service` is the owner of a long-lived resource. The owner controls
construction, liveness, and teardown through ``start()``, ``is_alive``, and
``shutdown()``. Its ``client()`` method returns the object intended for
consumers.

Remote clients are lightweight, picklable capabilities. They expose domain
operations such as policy calls, ``add`` and ``sample``, or ``log_scalar``, but
not lifecycle operations. A worker can use a service without being able to
terminate the process or actor shared by other workers.

.. code-block:: python

    owner = make_service()
    owner.start()

    client = owner.client()
    send_to_worker(client)

    # The owner remains in the driver.
    owner.shutdown()

Direct services make a deliberate exception to capability restriction:
``owner.client() is owner``. Adding a proxy in the same process would impose
overhead without creating a meaningful isolation boundary. Code that requires
restricted capabilities should therefore rely on that guarantee only for
remote backends.

The owner/client split also ensures that heavy resources are constructed once.
For example, a process logger constructs its concrete logging SDK inside the
logging process, and a Ray replay buffer constructs its storage and sampler in
its actor. Pickling a client does not reconstruct those resources.

Placement does not define communication
---------------------------------------

``service_backend`` selects where ownership lives. TorchRL uses the canonical
backend vocabulary ``direct``, ``thread``, ``process``, ``ray``, ``monarch``,
and ``distributed``, but each service supports only the placements that fit
its implementation.

Placement is intentionally separate from the operation protocol. The domains
have incompatible communication requirements:

* Inference is request/reply traffic that benefits from batching and
  specialized tensor transports.
* Replay buffers combine writes, sampling, priority updates, and storage
  ownership.
* Logging carries small scalars as well as large videos and must preserve SDK
  completion and error semantics.
* Weight synchronization distributes versioned model state rather than
  serving arbitrary requests.

Forcing these interactions behind one transport interface would either erase
important guarantees or fill the interface with operations that are
meaningless for most implementations. The common abstraction therefore covers
ownership and lifecycle; each domain retains its own communication contract.

.. list-table:: Service capabilities
   :header-rows: 1

   * - Domain
     - Owner placement
     - Worker-facing interface
   * - Inference
     - ``thread`` or ``process``; transport may use threads, process queues,
       slots, Ray, or Monarch
     - Callable TensorDict policy through
       :class:`~torchrl.modules.inference_server.PolicyClientModule`
   * - Logging
     - ``direct``, ``process``, or ``ray``
     - ``log_*`` methods
   * - Replay buffer
     - ``direct`` or ``ray``
     - ``add``, ``extend``, ``sample``, and priority updates

The supported combinations are explicit rather than emulated. For example, a
process deployment can place inference and logging in child processes while
keeping a replay buffer direct. This avoids presenting a process replay
backend whose performance and data-movement contract have not been defined.

Preserving domain APIs
----------------------

Consumers should not branch on service placement. A policy client remains a
TensorDict policy, a replay-buffer client remains a replay buffer, and a
logger client retains its logging methods. Only construction and lifecycle
belong to the owner.

:class:`~torchrl.modules.inference_server.PolicyClientModule` accepts an
inference owner, transport, or callable client and obtains the restricted
client automatically:

.. code-block:: python

    from torchrl.modules.inference_server import PolicyClientModule

    policy = PolicyClientModule(
        inference_owner,
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )
    replay_buffer = replay_owner.client()
    logger = logger_owner.client()

The resulting training code is independent of placement:

.. code-block:: python

    td = env.reset()
    for step in range(num_steps):
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

Moving an owner changes which calls cross an execution boundary; it does not
change the loop that consumes its client.

Completion and failure semantics
--------------------------------

Remote execution should not silently weaken the contract of a domain method.
TorchRL therefore uses acknowledged calls where their direct counterparts are
complete on return:

* Logger methods return after the concrete logger method has run. Service-side
  failures are raised at the call site, and custom ``log_*`` methods preserve
  their return values.
* Video logging waits for encoding or upload so evaluation cannot finish while
  its artifact is still pending. CUDA payloads are moved to CPU for transport,
  while video shape, dtype, and content are preserved.
* Replay-buffer operations return their result or raise the remote failure.
* A synchronous policy call returns the inference result; asynchronous
  inference remains available through its domain-specific submission API.

Acknowledgement adds a round trip to remote calls. For logging this favors
correctness, bounded memory, and immediate error reporting over fire-and-forget
throughput. Applications should avoid logging every hot-path intermediate and
prefer meaningful aggregated metrics.

Bounded queues and actor limits provide backpressure rather than allowing an
unbounded backlog. Clients preserve submission order for each producer;
independent producers do not imply a global ordering.

When a process or actor exits before replying, clients report peer failure
instead of waiting indefinitely. Startup also waits until the owned resource
is ready, so obtaining a usable owner implies that its service construction
has completed.

Lifecycle belongs to the owner
------------------------------

Clients never stop shared infrastructure. The driver shuts services down only
after collectors, evaluators, and other client users have finished. Explicit
teardown releases processes, actors, queues, feeder threads, and SDK resources
deterministically.

.. code-block:: python

    from contextlib import ExitStack

    with ExitStack() as services:
        logger_owner = make_logger()
        services.callback(logger_owner.shutdown)

        replay_owner = make_replay_buffer()
        services.callback(replay_owner.shutdown)

        inference_owner = make_inference_server()
        services.callback(inference_owner.shutdown)

        run_training(
            policy=PolicyClientModule(inference_owner),
            replay_buffer=replay_owner.client(),
            logger=logger_owner.client(),
        )

Callbacks run in reverse registration order, so consumers should be registered
after the services they use. Keeping the logger alive longest permits teardown
metrics to be recorded before its final flush and shutdown. Shutdown is
idempotent, which makes cleanup safe in both normal and exceptional paths.

Integrations accept owners when they can
----------------------------------------

An integration that only needs domain operations can accept either a logger
or logger owner and obtain the client internally. This keeps deployment
plumbing out of recipes. :class:`~torchrl.record.VideoRecorder`, for example,
accepts a logger owner directly:

.. code-block:: python

    from torchrl.record import VideoRecorder

    recorder = VideoRecorder(logger_owner, tag="eval/video", fps=30)

The recorder uses the restricted client and records vector-environment frames
as a synchronized grid. Lifecycle remains with ``logger_owner``.

Environments are execution resources, not shared services
---------------------------------------------------------

Environment instances carry trajectory state and require ordered, exclusive
stepping. Giving several interchangeable clients access to one environment
session would make reset and step ordering ambiguous. Their latency can also
be low enough that a generic remote call on every step dominates collection
cost.

TorchRL therefore scales environments through serial, parallel, or
asynchronous environment pools and through collectors that own environment
replicas. The actor loop is normally placed with its environment and uses a
policy client to reach shared inference when needed. This preserves session
affinity and allows environment communication to use specialized shared-memory
or asynchronous paths.

Remote simulators and physical systems still require networked environment
clients, but those clients need session leases, ordered step/reset semantics,
timeouts, and trajectory-aware recovery. Those requirements form an
environment-pool protocol rather than the generic shared-service contract.

Discovery is optional
---------------------

Explicitly passing clients is preferable when worker destinations are known at
construction time. Discovery is useful when independently created Ray workers
must locate a service by name. Registering a running owner stores its
restricted client without transferring ownership; removing the discovery
entry does not shut down that externally owned service.

See :ref:`ref_services` for registry and namespace behavior. Discovery does not
replace the owner/client lifecycle and does not grant workers shutdown rights.

Design compromises
------------------

The service model makes several trade-offs explicit:

* Direct clients use identity semantics for zero overhead and therefore do not
  provide capability isolation.
* Remote completion semantics add latency but prevent lost errors and preserve
  return values.
* Backend support differs by domain rather than exposing nominal combinations
  without a suitable transport and performance contract.
* Ownership is unified, while payload transport remains specialized for
  inference, logging, replay, commands, shared state, and weight updates.
* Environment execution remains a separate stateful abstraction.
* Discovery is opt-in and does not transfer lifecycle ownership.

These boundaries keep the consumer API small without hiding costs or weakening
domain guarantees.

Runnable examples
-----------------

The `service examples
<https://github.com/pytorch/rl/tree/main/examples/services>`_ demonstrate the
same TensorDict training loop with direct, process, and Ray placement. Their
README contains dependencies, commands, and the mapping from each profile to
its concrete owners.
