# Designing Training Applications with Services

Distributed reinforcement learning combines components with very different
resource and communication requirements. Actors need low-latency policy
inference, replay buffers coordinate concurrent producers and consumers, and
loggers own stateful SDKs, files, credentials, and artifact uploads. These
components often need to live in different threads, processes, or cluster
actors while remaining easy to use from the training loop.

Making an object remote is not sufficient. A useful distributed API must also
answer:

- Who constructs and owns the expensive resource?
- What can safely be copied or pickled into a worker?
- Which operations are available to workers, and which remain driver-only?
- When is an operation complete, and where are failures reported?
- Who shuts down child processes, actors, queues, and background threads?

TorchRL services answer these questions with a common ownership model while
preserving the domain API of each component.

## Owners and clients

A [`Service`](generated/torchrl.services.Service.html#torchrl.services.Service) is the owner of a long-lived resource. The owner controls
construction, liveness, and teardown through `start()`, `is_alive`, and
`shutdown()`. Its `client()` method returns the object intended for
consumers.

Remote clients are lightweight, picklable capabilities. They expose domain
operations such as policy calls, `add` and `sample`, or `log_scalar`, but
not lifecycle operations. A worker can use a service without being able to
terminate the process or actor shared by other workers.

```
owner = make_service()
owner.start()

client = owner.client()
send_to_worker(client)

# The owner remains in the driver.
owner.shutdown()
```

Direct services make a deliberate exception to capability restriction:
`owner.client() is owner`. Adding a proxy in the same process would impose
overhead without creating a meaningful isolation boundary. Code that requires
restricted capabilities should therefore rely on that guarantee only for
remote backends.

The owner/client split also ensures that heavy resources are constructed once.
For example, a process logger constructs its concrete logging SDK inside the
logging process, and a Ray replay buffer constructs its storage and sampler in
its actor. Pickling a client does not reconstruct those resources.

## Placement does not define communication

`service_backend` selects where ownership lives. TorchRL uses the canonical
backend vocabulary `direct`, `thread`, `process`, `ray`, `monarch`,
and `distributed`, but each service supports only the placements that fit
its implementation.

Placement is intentionally separate from the operation protocol. The domains
have incompatible communication requirements:

- Inference is request/reply traffic that benefits from batching and
specialized tensor transports.
- Replay buffers combine writes, sampling, priority updates, and storage
ownership.
- Logging carries small scalars as well as large videos and must preserve SDK
completion and error semantics.
- Weight synchronization distributes versioned model state rather than
serving arbitrary requests.

Forcing these interactions behind one transport interface would either erase
important guarantees or fill the interface with operations that are
meaningless for most implementations. The common abstraction therefore covers
ownership and lifecycle; each domain retains its own communication contract.

| Domain | Owner placement | Worker-facing interface |
| --- | --- | --- |
| Inference | `thread` or `process`; transport may use threads, process queues, slots, Ray, or Monarch | Callable TensorDict policy through [`PolicyClientModule`](generated/torchrl.modules.inference_server.PolicyClientModule.html#torchrl.modules.inference_server.PolicyClientModule) |
| Logging | `direct`, `process`, or `ray` | `log_*` methods |
| Replay buffer | `direct` or `ray` | `add`, `extend`, `sample`, and priority updates |

The supported combinations are explicit rather than emulated. For example, a
process deployment can place inference and logging in child processes while
keeping a replay buffer direct. This avoids presenting a process replay
backend whose performance and data-movement contract have not been defined.

## Choosing a payload transport

Ray can be used for ownership without requiring TensorDict payloads to travel
through Ray. With `service_backend="ray"`, Ray creates, places, monitors, and
stops the service actor. The separate `transport` argument selects how
clients exchange payloads with that actor:

- `transport="ray"` uses Ray calls, queues, and the Ray object store. It is
the flexible option for changing layouts and Python or non-tensor values.
- `transport="distributed"` uses standalone PyTorch distributed
point-to-point groups for fixed-layout TensorDict payloads. Gloo carries CPU
tensors and NCCL carries CUDA tensors.
- `transport="auto"` selects the placement default. For a Ray-owned service
it currently resolves to `"ray"`; it does not inspect the payload or
silently switch transports later.

Ray remains the control and placement plane when the distributed payload path
is selected. Transport-owned process groups do not replace or destroy the
application's default process group. The current distributed implementation
is point-to-point request/reply. Collective data movement can be added behind
the same selector when a component has a collective communication pattern.

The supported combinations are listed below. An em dash means that the
combination is rejected during construction rather than falling back to a
different transport.

| Component | `service_backend` | `auto` | `ray` | `distributed` | Other supported transports |
| --- | --- | --- | --- | --- | --- |
| [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) | `direct` | `direct` | - | - | `direct` |
| [`RayReplayBuffer`](generated/torchrl.data.RayReplayBuffer.html#torchrl.data.RayReplayBuffer) or a replay buffer constructed with `service_backend="ray"` | `ray` | `ray` | Yes | Gloo or NCCL | - |
| [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) | `thread` | `thread` | - | - | `thread`, `queue`, or `direct` |
| [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) | `process` | `process` | - | Gloo or NCCL [1] | `process` or `shared_memory` |
| [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) | `ray` | `ray` | Yes | Gloo or NCCL | - |

### Transport characteristics

Performance depends on payload size, request rate, topology, and batching.
The following table describes expected trade-offs rather than promising a
universal ordering.

| Transport | Layout | Payload values | Device | Expected behavior |
| --- | --- | --- | --- | --- |
| `thread` / `direct` | Dynamic | Values accepted by the component | Same process | Lowest communication overhead. There is no serialization boundary; batching and model execution usually dominate. |
| `process` | Dynamic | Pickle-compatible values, including non-tensor data | Multiprocessing-supported | Flexible, but queueing and serialization make large or frequent payloads more expensive than a preallocated tensor path. |
| `shared_memory` | Fixed keys, shapes, dtypes, and batch sizes | Tensor leaves only | CPU | Preallocated slots avoid pickling tensor contents. This is generally a better process-local choice for large, stable CPU TensorDict payloads. |
| `ray` | Dynamic | Ray-serializable values, including strings and non-tensor data | Ray-supported | Easiest remote option and the compatibility fallback. Ray RPC, queue, and object-store overhead can dominate small, frequent RL messages. |
| `distributed` with Gloo | Fixed keys, shapes, dtypes, devices, and batch sizes | Tensor leaves only | CPU | Avoids putting steady-state tensor payloads through Ray. It is usually preferable for stable CPU TensorDict traffic once rendezvous and group setup costs are amortized. |
| `distributed` with NCCL | Fixed keys, shapes, dtypes, devices, and batch sizes | Tensor leaves only | CUDA | Sends CUDA tensors without CPU staging. It is intended for stable, sufficiently large GPU payloads; setup and per-message latency may outweigh the benefit for small requests. Assign each NCCL rank a distinct GPU in normal deployments. Colocating ranks on one GPU is supported by NCCL only when `NCCL_MULTI_RANK_GPU_ENABLE=1` is set in every participating process. |

Distributed layouts are bound to an endpoint generation. Extra TensorDict
keys are not transmitted, while a missing declared key or a later incompatible
shape, dtype, device, or replay sample batch size raises an error. There is no
automatic fallback to Ray for an incompatible payload. Choose `"ray"` when
the data contains strings, [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData), arbitrary Python
objects, or genuinely changing tensor layouts.

### Ray collectors and service transports

[`RayCollector`](generated/torchrl.collectors.distributed.RayCollector.html#torchrl.collectors.distributed.RayCollector) deliberately has no
`transport` argument. A transport belongs to the endpoint that receives and
serves the data:

- policy requests use the attached inference server's transport;
- replay writes use the attached replay buffer's transport; and
- Ray continues to schedule and control the collector actors.

When `replay_buffer` is provided, each collector worker receives its own
restricted replay endpoint and writes its rollout directly to replay. The
worker returns only completion to the driver, so the rollout is not copied
through Ray a second time. This is the recommended high-throughput topology:

```
from functools import partial

from torchrl.collectors.distributed import RayCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules.inference_server import InferenceServer

replay = TensorDictReplayBuffer(
 storage=partial(LazyTensorStorage, 100_000),
 batch_size=256,
 service_backend="ray",
 service_backend_options={"remote_config": {"num_cpus": 1}},
 transport="distributed",
 transport_options={"backend": "gloo"},
)
inference = InferenceServer(
 policy_factory=make_policy,
 service_backend="ray",
 service_backend_options={"remote_config": {"num_cpus": 1}},
 transport="distributed",
 transport_options={"backend": "gloo"},
)
collector = RayCollector(
 create_env_fn=[make_env] * 8,
 policy=inference,
 replay_buffer=replay,
 frames_per_batch=1024,
)
```

If no replay buffer is attached, iterating over `RayCollector` returns
rollouts to the driver through Ray's object store. That collector-to-driver
path is not transport-selectable. Use direct-to-replay collection when the
rollouts are large and do not need to be inspected by the driver. Use the Ray
service transport instead of the distributed transport when the policy or
replay payload contains non-tensor or dynamic data.

## Preserving domain APIs

Consumers should not branch on service placement. A policy client remains a
TensorDict policy, a replay-buffer client remains a replay buffer, and a
logger client retains its logging methods. Only construction and lifecycle
belong to the owner.

[`PolicyClientModule`](generated/torchrl.modules.inference_server.PolicyClientModule.html#torchrl.modules.inference_server.PolicyClientModule) accepts an
inference owner, transport, or callable client and obtains the restricted
client automatically:

```
from torchrl.modules.inference_server import PolicyClientModule

policy = PolicyClientModule(
 inference_owner,
 in_keys=["observation"],
 out_keys=["action", "policy_version"],
)
replay_buffer = replay_owner.client()
logger = logger_owner.client()
```

The resulting training code is independent of placement:

```
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
```

Moving an owner changes which calls cross an execution boundary; it does not
change the loop that consumes its client.

## Completion and failure semantics

Remote execution should not silently weaken the contract of a domain method.
TorchRL therefore uses acknowledged calls where their direct counterparts are
complete on return:

- Logger methods return after the concrete logger method has run. Service-side
failures are raised at the call site, and custom `log_*` methods preserve
their return values.
- Video logging waits for encoding or upload so evaluation cannot finish while
its artifact is still pending. CUDA payloads are moved to CPU for transport,
while video shape, dtype, and content are preserved.
- Replay-buffer operations return their result or raise the remote failure.
- A synchronous policy call returns the inference result; asynchronous
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

## Lifecycle belongs to the owner

Clients never stop shared infrastructure. The driver shuts services down only
after collectors, evaluators, and other client users have finished. Explicit
teardown releases processes, actors, queues, feeder threads, and SDK resources
deterministically.

```
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
```

Callbacks run in reverse registration order, so consumers should be registered
after the services they use. Keeping the logger alive longest permits teardown
metrics to be recorded before its final flush and shutdown. Shutdown is
idempotent, which makes cleanup safe in both normal and exceptional paths.

## Integrations accept owners when they can

An integration that only needs domain operations can accept either a logger
or logger owner and obtain the client internally. This keeps deployment
plumbing out of recipes. [`VideoRecorder`](generated/torchrl.record.VideoRecorder.html#torchrl.record.VideoRecorder), for example,
accepts a logger owner directly:

```
from torchrl.record import VideoRecorder

recorder = VideoRecorder(logger_owner, tag="eval/video", fps=30)
```

The recorder uses the restricted client and records vector-environment frames
as a synchronized grid. Lifecycle remains with `logger_owner`.

## Environments are execution resources, not shared services

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

## Discovery is optional

Explicitly passing clients is preferable when worker destinations are known at
construction time. Discovery is useful when independently created Ray workers
must locate a service by name. Registering a running owner stores its
restricted client without transferring ownership; removing the discovery
entry does not shut down that externally owned service.

See [Service owners and clients](services.html#ref-services) for registry and namespace behavior. Discovery does not
replace the owner/client lifecycle and does not grant workers shutdown rights.

## Design compromises

The service model makes several trade-offs explicit:

- Direct clients use identity semantics for zero overhead and therefore do not
provide capability isolation.
- Remote completion semantics add latency but prevent lost errors and preserve
return values.
- Backend support differs by domain rather than exposing nominal combinations
without a suitable transport and performance contract.
- Ownership is unified, while payload transport remains specialized for
inference, logging, replay, commands, shared state, and weight updates.
- Environment execution remains a separate stateful abstraction.
- Discovery is opt-in and does not transfer lifecycle ownership.

These boundaries keep the consumer API small without hiding costs or weakening
domain guarantees.

## Runnable examples

The [service examples](https://github.com/pytorch/rl/tree/main/examples/services) demonstrate the
same TensorDict training loop with direct, process, and Ray placement. Their
README contains dependencies, commands, and the mapping from each profile to
its concrete owners.

## Distributed transport implementation notes

The distributed transport does not negotiate arbitrary Python values on every
request. Instead, each request/reply channel is constructed from representative
TensorDicts that define a static wire layout. The layout includes the sorted
nested keys and each tensor leaf's shape, dtype, and device. Gloo layouts must
contain CPU tensors and NCCL layouts must contain CUDA tensors. Non-tensor
leaves are rejected.

### How layouts are established

Layout establishment depends on the service owner:

| Owner | Source of the layout | Bootstrap behavior |
| --- | --- | --- |
| Process-owned inference | Explicit `request_spec` and `response_spec` arguments | Both layouts must be available before the subprocess starts. There is no Ray control channel through which to inspect a first request. |
| Ray-owned inference | Explicit arguments, or the first request when they are omitted | For lazy discovery, the first request reaches the inference actor through Ray. The actor applies the configured collation, device, interaction-type, policy, output-device, and policy-version rules to a cloned request. The resulting request and response establish the distributed channel, after which the original request is submitted through that channel. |
| Ray-owned replay | Explicit operation specs, or the first call to each operation | Replay has independent layouts for `extend`, `sample`, and priority updates. The first call is executed by the replay actor through Ray, its actual result is returned to the caller, and its payload and result establish that operation's channel. The control layout used by `len()` and `write_count` is known in advance. |

Lazy inference discovery performs a probe policy forward to determine the
response layout, followed by the user-visible forward through the newly
created distributed channel. A policy whose forward mutates state should use
explicit `request_spec` and `response_spec` values to avoid probe side
effects. Replay bootstrap operations are not repeated: the operation executed
through Ray is the caller's first operation.

### Request size versus server batch size

Inference has two distinct batching dimensions. The *request size* is the
batch shape inside one TensorDict submitted by one client. The *server batch
size* is the number of queued requests selected for one policy forward. The
distributed transport fixes the per-request layout, but it does not fix the
number of requests combined by the server.

For example, suppose the discovered request contains an observation with
shape `[4]` and no TensorDict batch dimensions. The transport always moves
requests with that layout. The server may nevertheless drain anywhere from
one to `max_batch_size` such requests, stack them into an observation tensor
with shape `[num_requests, 4]`, run the policy, and unbind the leading
request dimension so that each client receives one response through its own
fixed response buffer. Queue timing therefore continues to determine the
effective policy-forward batch size.

If a caller instead submits a TensorDict with batch size `[N]`, then `N`
is part of the distributed request and response layouts. Later calls through
that endpoint must use the same `N`. Applications with genuinely changing
per-request batch shapes should pad or split requests to a fixed layout, or
use the Ray transport. Dynamic transports remove the static wire-layout
restriction, although the configured `collate_fn` and policy must still be
able to combine the submitted requests. `max_batch_size` counts queued
requests, not the number of samples contained inside them.

Replay binds its sample layout to the batch size of the first sample. Extend
and sample layouts may instead be supplied with
`transport_options["extend_spec"]`, `transport_options["sample_spec"]`,
respectively. An explicit sample layout also seeds the priority-update layout;
`transport_options["priority_spec"]` can override it when supplied alongside
`sample_spec`. Explicit inference layouts use the top-level `request_spec`
and `response_spec` arguments.

### Buffers and the steady-state path

The representative TensorDicts are normalized to contiguous tensors and kept
as layout templates. Each client obtains an independent endpoint. When that
endpoint connects for the first time, the client and owner create their own
request and response staging buffers from the templates, along with separate
point-to-point process groups for each direction. Small request identifiers,
statuses, and exception notifications use Gloo control groups. Tensor leaves
use the selected Gloo or NCCL payload groups in deterministic key order.

For each subsequent request, TorchRL selects the declared keys with strict
missing-key checking, copies them into the existing send buffer, and sends its
tensor leaves. The receiver fills its existing receive buffer. The current
implementation then clones a received request before queueing it and clones a
received response before handing it to a waiting caller, because the staging
buffers are reused immediately. The transport therefore avoids per-request
payload serialization and wire-buffer allocation, but it is not completely
allocation-free.

### Layout lifetime and validation

A discovered layout belongs to one endpoint generation and never changes in
place. Extra TensorDict keys are not transmitted. Missing declared keys, or
incompatible shapes, dtypes, devices, and replay sample batch sizes, raise an
error rather than renegotiating or falling back to Ray. Restarting an owner
creates a new generation with new endpoints, rendezvous state, process groups,
and layouts.

These mechanisms are private implementation details rather than importable
service APIs. The main implementation boundaries are
`torchrl._comm.distributed` for static request/reply communication,
`torchrl._comm.replay_service` for replay operation channels, and
`torchrl.modules.inference_server._server` for inference ownership and lazy
Ray bootstrap. They may be refactored without changing the public
`service_backend` and `transport` selectors.