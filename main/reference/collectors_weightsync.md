# Weight Synchronization

RL pipelines are typically split in two big computational buckets: training, and inference.
While the inference pipeline sends data to the training one, the training pipeline needs to occasionally
synchronize its weights with the inference one.
In the most basic setting (fully synchronized data collection with traditional neural networks), the same weights are
used in both instances. From there, anything can happen:

- In multiprocessed or distributed settings, several copies of the policy can be held by the inference workers (named
DataCollectors in TorchRL). When synchronizing the weights, each worker needs to receive a new copy of the weights
for their instance of the policy.
- In some cases, the environment or the postprocessing hooks can rely on the usage of a model which itself needs
synchronization. This means that there can be multiple ends in the data transfer API and one needs to think beyond
policy-to-policy weight synchronization strategies.
- In the LLM world, the inference engine and the training one are very different: they will use different libraries,
kernels and calling APIs (e.g., generate vs. forward). The weight format can also be drastically different (quantized
vs non-quantized).
This makes the weight synchronization much more complex, as one cannot simply dump and load a state dict on both ends.
- One typically also has to choose who instantiates a transfer: should this come from the inference engine who actively
asks for new weights, or must it only be the trainer who pushes its weights to the workers? An intermediate approach
is to store the weights on some intermediary server and let the workers fetch them when necessary.

TorchRL tries to account for each of these problems in a flexible manner. We identify three basic components in a weight
transfer:

- A **Scheme** class that orchestrates the entire weight synchronization lifecycle, including initialization,
connection setup, and weight transfer coordination.
- A **Transport** class that handles the actual transfer of weights (through shared memory, queues, torch.distributed,
Ray, etc.). Each scheme creates one or more transports for communication with workers.
- A **Strategy** class that determines the weight format (TensorDict or state_dict) and how weights are
extracted from and applied to models.

Each of these classes is detailed below.

Note

**For most users, weight synchronization happens automatically.** When using TorchRL collectors
with the `weight_sync_schemes` argument, the collector handles all initialization, connection,
and synchronization calls internally. You simply call `collector.update_policy_weights_()` and
the weights are propagated to all workers.

The `update_policy_weights_` method supports multiple calling conventions:

```
# No arguments - uses registered policy
collector.update_policy_weights_()

# Positional argument - policy module or TensorDict
collector.update_policy_weights_(policy_module)
collector.update_policy_weights_(weights_tensordict)

# Keyword arguments for clarity
collector.update_policy_weights_(policy=actor_module)
collector.update_policy_weights_(weights=weights_td, model_id="actor")

# Multiple models atomically
collector.update_policy_weights_(weights_dict={"actor": actor_td, "critic": critic_td})
```

The detailed lifecycle documentation below is primarily intended for developers who want to:

- Understand the internals of weight synchronization
- Implement custom weight sync schemes for specialized use cases (e.g., new distributed backends, custom serialization)
- Debug synchronization issues in complex distributed setups
- Use weight sync schemes outside of collectors for custom multiprocessing scenarios

## Lifecycle of Weight Synchronization

Weight synchronization follows a **two-phase initialization pattern** with a clear separation between
local setup and inter-process communication.

For **queue / store-based schemes** (e.g. multiprocessing, TCPStore), the receiver starts a small
**background loop** that waits for "update" instructions and runs the actual receive/apply logic.

For **RPC / Ray schemes**, the sender triggers the receiver via a **remote call** to
`_receive_weights_scheme()`, which runs `scheme.receive()` on the receiver side (no dedicated
background thread is required).

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SENDER (Main Process) │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. scheme.init_on_sender(model_id, context, ...) │
│ └─ Sets up local state, creates transports, NO communication │
│ │
│ 2. Make scheme available on receiver (scheme-dependent) │
│ └─ e.g. via multiprocessing pickle/serialization, RPC, Ray actor init │
│ │
│ 3. scheme.connect() ◄──── BLOCKING RENDEZ-VOUS ────► │
│ └─ Sets up connection / rendez-vous │
│ └─ May send initial weights (scheme-dependent) │
│ │
│ 4. scheme.send(weights) [ready for ongoing updates] │
│ └─ Triggers receiver to run ``scheme.receive()`` │
│ (instruction queue / TCPStore / remote call, scheme-dependent) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ RECEIVER (Worker Process) │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. scheme.init_on_receiver(model_id, context, ...) │
│ └─ Sets up local state, resolves model, NO communication │
│ │
│ 2. scheme.connect() ◄──── BLOCKING RENDEZ-VOUS ────► │
│ └─ Receives initial weights (scheme-dependent) │
│ └─ If needed: starts a background loop for update instructions │
│ │
│ 3. Receiver-side handler (scheme-dependent) │
│ └─ Background thread for queue/store schemes │
│ └─ RPC/Ray remote call handler for RPC/Ray schemes │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Initialization (No Communication)

The `init_on_sender()` and `init_on_receiver()` methods prepare local state without any
inter-process communication:

- Set up local attributes and references (model, context, worker indices)
- Create transport objects and register them
- Prepare queues, buffers, or other communication primitives
- **Do NOT perform any inter-worker communication**

This separation allows the scheme to be pickled and sent to worker processes after sender
initialization but before any actual communication occurs.

```
# === SENDER (main process) ===
scheme = SharedMemWeightSyncScheme()
scheme.init_on_sender(
 model_id="policy",
 context=collector, # or explicit params like weights, devices, num_workers
)

# === Scheme is passed to workers via multiprocessing ===
# (The scheme object is pickled and sent to worker processes)

# === RECEIVER (worker process) ===
scheme.init_on_receiver(
 model_id="policy",
 context=inner_collector, # or explicit params like model, worker_idx
)
```

### Phase 2: Connection and Initial Weights (Rendez-vous)

The `connect()` method performs the actual inter-process communication. **In most schemes, both
sender and receiver call this method** (simultaneously or in the expected order for the scheme).
Some specialized schemes can be sender-driven (e.g. `RayModuleTransformScheme` triggers receiver setup
via a Ray call).

1. **Connection rendez-vous**: Sender and receiver synchronize (e.g., torch.distributed process group
initialization, shared memory buffer exchange via queues)
2. **Initial weight transfer** (scheme-dependent): Some schemes send initial weights during `connect()`
(e.g. `SharedMemWeightSyncScheme`, `MultiProcessWeightSyncScheme`, `DistributedWeightSyncScheme`,
`RayWeightSyncScheme`). Others (notably `RPCWeightSyncScheme`) typically start synchronizing on the
first `send()` call.
3. **Receiver readiness**: For queue/store-based schemes, `connect()` starts a background loop on the
receiver that waits for update instructions.

```
# === Called simultaneously on both ends ===

# Sender side (main process):
scheme.connect() # Blocks until rendez-vous completes (scheme-dependent)

# Receiver side (worker process):
scheme.connect(worker_idx=0) # Blocks until rendez-vous completes (scheme-dependent)
```

Note

The `connect()` method is a **blocking rendez-vous** for most schemes. The exact behavior
depends on the scheme:

- **Queue-based schemes** (SharedMem, MultiProcess): Sender puts to queue, receiver blocks reading
- **Distributed schemes** (Distributed, Ray): Both sides block on `torch.distributed.send/recv`
- **RPC/Ray with remote calls**: Receiver's `connect()` may be a no-op if the sender triggers
the receiver via a remote call (e.g., `RayModuleTransformScheme`)

### Phase 3: Ongoing Weight Updates

After `connect()` completes, the scheme is ready for ongoing weight synchronization. The sender
calls `send()` / `send_async()` to push weights and trigger the receiver to run `scheme.receive()`.

```
# Training loop
for batch in dataloader:
 loss = train_step(batch)
 scheme.send(new_weights)
```

## Scheme-Specific Behavior

### SharedMemWeightSyncScheme

Uses shared memory for zero-copy weight updates. After initial setup, weight updates are instantaneous
since all processes share the same memory buffers.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates shared buffers + instruction queues | Stores model reference | None |
| `connect` | Sends buffer references + initial weights | Receives buffers, applies weights, starts background thread | mp.Queue (blocking) |
| `send` | Updates shared memory, sends instruction | Background thread applies shared memory weights | Zero-copy shared memory + mp.Queue |

### MultiProcessWeightSyncScheme

Sends weight copies through multiprocessing queues. More flexible than shared memory but requires
explicit data transfer for each update. Supports timeout for non-blocking receives.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates weight + instruction queues | Gets queue references | None |
| `connect` | Sends initial weights | Receives weights, applies via strategy, starts background thread | mp.Queue (blocking) |
| `send` | Puts weights + instruction | Background thread receives and applies weights | mp.Queue (supports timeout) |

### DistributedWeightSyncScheme

Uses `torch.distributed` primitives with a TCPStore for signaling. Suitable for distributed
training scenarios where processes are already part of a process group. Supports timeout via
`irecv(return_premature=True)` for non-blocking receives.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates transports with TCPStore + rank | Creates transport with store + rank | None |
| `connect` | Sends initial weights via `torch.distributed.send()` | Receives weights, applies via strategy, starts background thread | torch.distributed send/recv |
| `send` | Sets TCPStore flag + `torch.distributed.send()` | Background thread polls TCPStore and receives weights | TCPStore + torch.distributed (supports timeout) |

### RPCWeightSyncScheme

Uses `torch.distributed.rpc` for signaling with `torch.distributed` for data transfer.
The sender's transport signals the remote collector via an RPC call to `_receive_weights_scheme()`,
and then transfers weights via `torch.distributed` send/recv. Supports timeout via
`irecv(return_premature=True)` for non-blocking receives.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates transports with RPC refs | Stores model reference, creates transport | None |
| `connect` | No-op for RPC transport (no initial weight transfer) | No-op | None |
| `send` | RPC call to `_receive_weights_scheme()` + `torch.distributed.send()` | Receiver runs `scheme.receive()` in the RPC call context and applies weights | RPC + torch.distributed (supports timeout) |

### RayWeightSyncScheme

Uses Ray actors for coordination with `torch.distributed` for efficient weight transfer.
Suitable for Ray-based distributed RL setups. Supports timeout via `irecv(return_premature=True)`
for non-blocking receives.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates transports with Ray actor handles | Creates transport, stores model | None |
| `connect` | Creates ConnectionInfo, `init_process_group(rank=0)`, sends initial weights | Waits for ConnectionInfo, `init_process_group(rank=N)`, receives weights | Ray actor + torch.distributed |
| `send` | Ray remote call to `_receive_weights_scheme()` + `torch.distributed.isend()` | Receiver runs `scheme.receive()` in the Ray call context and applies weights | Ray + torch.distributed (supports timeout) |

### RayModuleTransformScheme

Specialized scheme for synchronizing weights to a module running inside a `RayModuleTransform`.
The sender triggers all receiver operations via Ray remote calls.

| Phase | Sender | Receiver | Communication |
| --- | --- | --- | --- |
| `init` | Creates transport for transform actor | Creates transport, stores module | None |
| `connect` | Ray call triggers receiver init + weight send | Triggered by Ray: joins process group, receives weights | Ray + torch.distributed |
| `send` | Ray remote call to `_receive_weights_scheme()` + `torch.distributed.isend()` | Receiver runs `scheme.receive()` in the Ray call context and applies weights | Ray + torch.distributed |

Note

`RayModuleTransformScheme` is unique in that even `connect` on the sender
triggers the receiver initialization via a Ray remote call. The user only needs to call
`connect()` on the sender side.

### Background Thread Architecture

Some schemes use a **background receiver thread** on the receiver side. This is used when the sender
cannot directly invoke receiver logic (e.g. multiprocessing queues or TCPStore-based signaling).
The thread is started during `connect()` and runs `scheme.receive()` when instructed by the sender.

**Instruction mechanisms** (scheme-specific):
- **SharedMem/MultiProcess**: Queue-based (`queue.put("receive")`)
- **Distributed**: TCPStore-based (`store.set("receive")`)
- **RPC/Ray**: Remote calls to `_receive_weights_scheme()` (no dedicated background thread)

**Benefits**: non-blocking main process for queue/store-based schemes, sender-triggered updates,
automatic cascading to sub-collectors, and graceful timeout handling.

## Usage Examples

Note

**Runnable versions** of these examples are available in the repository:

- [examples/collectors/weight_sync_standalone.py](https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_standalone.py): Standalone weight synchronization
- [examples/collectors/weight_sync_collectors.py](https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_collectors.py): Collector integration
- [examples/collectors/multi_weight_updates.py](https://github.com/pytorch/rl/blob/main/examples/collectors/multi_weight_updates.py): Multi-model weight sync (policy + env + replay buffer)

### Using Weight Sync Schemes with Collectors

Weight sync schemes integrate seamlessly with TorchRL collectors. The collector handles calling
`init_on_sender()`, `init_on_receiver()`, and `connect()` automatically:

```
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import MultiCollector
from torchrl.envs import GymEnv
from torchrl.weight_update import SharedMemWeightSyncScheme

# Create environment and policy
env = GymEnv("CartPole-v1")
policy = TensorDictModule(
 nn.Linear(env.observation_spec["observation"].shape[-1],
 env.action_spec.shape[-1]),
 in_keys=["observation"],
 out_keys=["action"],
)

# Create scheme - collector handles initialization
scheme = SharedMemWeightSyncScheme(strategy="tensordict")

collector = MultiCollector(
 sync=True,
 create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
 policy=policy,
 frames_per_batch=192,
 total_frames=10000,
 weight_sync_schemes={"policy": scheme},
)

# Collect data and update weights
for i, data in enumerate(collector):
 # ... training step ...

 # Update weights - multiple calling conventions supported:
 if i % 10 == 0:
 # Option 1: No arguments (uses registered policy)
 collector.update_policy_weights_()

 # Option 2: Pass policy module (positional)
 collector.update_policy_weights_(policy)

 # Option 3: Pass weights TensorDict (positional)
 # collector.update_policy_weights_(weights_tensordict)

 # Option 4: Use keyword arguments for clarity
 # collector.update_policy_weights_(policy=policy)
 # collector.update_policy_weights_(weights=weights_td, model_id="policy")

collector.shutdown()
```

### Using Weight Sync Schemes Standalone

For custom multiprocessing scenarios, you can use schemes directly. The key is to follow the
two-phase pattern: initialize first (no communication), then connect (blocking rendez-vous):

```
import torch
import torch.nn as nn
from torch import multiprocessing as mp
from tensordict import TensorDict
from torchrl.weight_update import SharedMemWeightSyncScheme

def worker_fn(scheme, worker_idx):
 """Worker process - receives scheme via pickle."""
 # Create local model (weights will be overwritten by sender's weights)
 model = nn.Linear(4, 2)

 # PHASE 1: Initialize on receiver (no communication yet)
 scheme.init_on_receiver(model_id="policy", model=model, worker_idx=worker_idx)

 # PHASE 2: Blocking rendez-vous - receive initial weights from sender
 scheme.connect(worker_idx=worker_idx)
 # model now has the sender's weights; background thread started

 # Ready to work - background thread handles weight updates automatically
 while True:
 # ... use model for inference ...

# === MAIN PROCESS (Sender) ===
policy = nn.Linear(4, 2)
scheme = SharedMemWeightSyncScheme()

# PHASE 1: Initialize on sender (no communication yet)
scheme.init_on_sender(
 model_id="policy",
 weights=TensorDict.from_module(policy),
 devices=[torch.device("cpu")] * 2,
 num_workers=2,
)

# Spawn workers - scheme is pickled and sent to each worker
workers = [mp.Process(target=worker_fn, args=(scheme, i)) for i in range(2)]
for w in workers:
 w.start()

# PHASE 2: Blocking rendez-vous - send initial weights to workers
scheme.connect()
# Workers now have copies of policy's weights!

# PHASE 3: Ongoing updates (zero-copy for shared memory)
for epoch in range(10):
 # ... training step updates policy weights ...
 scheme.send() # Background threads automatically apply weights

scheme.shutdown() # Stop background threads
for w in workers:
 w.join()
```

Note

With `SharedMemWeightSyncScheme`, weight updates are zero-copy since all processes share the same
memory buffers. Background threads automatically apply updates when instructed by the sender.

Note

The `strategy` parameter determines the weight format: `"state_dict"` uses PyTorch's native state
dictionaries, while `"tensordict"` (default) uses TensorDict format which is more efficient for
structured models and supports features like device mapping.

### Multi-Model Weight Sync in Collectors

In many RL pipelines, the policy is not the only model that needs
synchronization. Common cases include:

- **Env transforms** such as [`ModuleTransform`](generated/torchrl.envs.transforms.ModuleTransform.html#torchrl.envs.transforms.ModuleTransform)
wrapping a learned feature extractor.
- **Replay buffer transforms** with a module (e.g., a learned priority model
or a preprocessing network).
- **Running statistics** from [`VecNormV2`](generated/torchrl.envs.transforms.VecNormV2.html#torchrl.envs.transforms.VecNormV2)
(running mean, variance, count).

The `weight_sync_schemes` dict maps **model IDs** (dotted paths into the
collector's object tree) to scheme instances. The `update_policy_weights_`
method then accepts a `weights_dict` to update all models atomically.

Note

**Runnable version**: [examples/collectors/multi_weight_updates.py](https://github.com/pytorch/rl/blob/main/examples/collectors/multi_weight_updates.py)

#### Model ID Paths

Model IDs are dotted attribute paths resolved from the collector.
Common patterns:

| Model ID | Resolves to |
| --- | --- |
| `"policy"` | The collector's policy module |
| `"env.transform[0]"` | First transform on the env |
| `"env.transform[0].module"` | The `module` attribute of a [`ModuleTransform`](generated/torchrl.envs.transforms.ModuleTransform.html#torchrl.envs.transforms.ModuleTransform) |
| `"replay_buffer.transform[0].module"` | Module inside a replay-buffer transform |

Subscript notation (`[0]`, `["key"]`) is supported for indexing into
sequences or dicts.

#### Example: Policy + Env Transform + Replay Buffer

```
from torchrl.collectors import MultiSyncCollector
from torchrl.weight_update import MultiProcessWeightSyncScheme

weight_sync_schemes = {
 "policy": MultiProcessWeightSyncScheme(strategy="state_dict"),
 "env.transform[0].module": MultiProcessWeightSyncScheme(
 strategy="tensordict"
 ),
 "replay_buffer.transform[0].module": MultiProcessWeightSyncScheme(
 strategy="tensordict"
 ),
}

collector = MultiSyncCollector(
 create_env_fn=[make_env, make_env],
 policy_factory=policy_factory,
 frames_per_batch=200,
 total_frames=10000,
 weight_sync_schemes=weight_sync_schemes,
 replay_buffer=rb,
)

for data in collector:
 # ... training step ...

 # Atomic update of all three models
 collector.update_policy_weights_(
 weights_dict={
 "policy": train_policy,
 "env.transform[0].module": train_env_module,
 "replay_buffer.transform[0].module": train_rb_module,
 }
 )

collector.shutdown()
```

#### VecNormV2 Running Statistics

[`VecNormV2`](generated/torchrl.envs.transforms.VecNormV2.html#torchrl.envs.transforms.VecNormV2) stores running mean, variance,
and count as "extra state" (via PyTorch's `get_extra_state()` /
`set_extra_state()`). These are **not** captured by
`TensorDict.from_module()` (which only sees `nn.Parameter` and
registered buffers), but **are** captured by both:

- The `"state_dict"` strategy (via PyTorch's `state_dict()`).
- The `"tensordict"` strategy, which additionally calls
`get_extra_state()` and stores the result under a special
`"__extra_state__"` key in the TensorDict.

This means no special handling is needed for the *transfer* -- just
register a scheme for the VecNormV2 transform and include it in
`weights_dict`.

**Freezing worker-side VecNormV2**: when collector workers receive
stats from the trainer, they should **not** update those stats with
their own data. Freeze VecNormV2 in the worker env factory so that
only the training process accumulates statistics:

```
from torchrl.collectors import MultiSyncCollector
from torchrl.weight_update import MultiProcessWeightSyncScheme

def make_worker_env():
 """Worker env: VecNormV2 is frozen so stats come from the trainer."""
 env = make_base_env() # includes VecNormV2 transform
 # Freeze so the worker doesn't update running stats locally
 for t in env.transform:
 if hasattr(t, "freeze"):
 t.freeze()
 return env

weight_sync_schemes = {
 "policy": MultiProcessWeightSyncScheme(strategy="tensordict"),
 "env.transform[0]": MultiProcessWeightSyncScheme(
 strategy="tensordict"
 ),
}

collector = MultiSyncCollector(
 create_env_fn=[make_worker_env, make_worker_env],
 policy_factory=policy_factory,
 frames_per_batch=200,
 total_frames=10000,
 weight_sync_schemes=weight_sync_schemes,
)

for data in collector:
 # ... training step (updates train_env VecNormV2 stats) ...

 # Push trainer's stats to frozen worker VecNormV2 instances
 collector.update_policy_weights_(
 weights_dict={
 "policy": train_policy,
 "env.transform[0]": train_env.transform[0], # VecNormV2
 }
 )

collector.shutdown()
```

Important

The `"tensordict"` strategy captures `get_extra_state()` from
**any** module that defines it, not just VecNormV2. If you have
custom modules with stateful buffers exposed via this PyTorch
protocol, they will be synchronized automatically.

Note

The [`Evaluator`](generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator) **automatically freezes**
VecNormV2 transforms in the eval env (see VecNormV2 Running Stats
below). For regular collectors, you must freeze explicitly in the
env factory as shown above.

## Evaluator Weight Sync

The [`Evaluator`](generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator) supports multi-model weight
synchronization through two complementary mechanisms:

1. **``weights_dict``** parameter on `evaluate()` / `trigger_eval()` -
a dict mapping model IDs to weight sources (`nn.Module` or
`TensorDictBase`).
2. **``weight_sync_schemes``** parameter on `Evaluator.__init__()` -
enables process-level isolation via
[`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) (1 worker) with
scheme-based weight transfer.

### Thread Backend (Same Process)

Without `weight_sync_schemes`, the evaluator runs in the same process
(thread backend) and applies weights directly via `to_module()`. The
`weights_dict` parameter lets you sync multiple models:

```
from torchrl.collectors import Evaluator

evaluator = Evaluator(env, policy, max_steps=1000)

# Sync policy only (backward-compatible)
evaluator.evaluate(weights=train_policy, step=0)

# Sync policy + env transform (e.g., VecNormV2)
evaluator.evaluate(
 weights_dict={
 "policy": train_policy,
 "env.transform[0]": train_env.transform[0],
 },
 step=0,
)
```

### Process Backend (CUDA Isolation)

With `backend="process"` or `weight_sync_schemes`, the evaluator
creates a [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) with a single
worker process internally. This provides full CUDA context isolation
and uses the weight-sync-scheme infrastructure for cross-process weight
transfer:

```
from torchrl.collectors import Evaluator
from torchrl.weight_update import MultiProcessWeightSyncScheme

evaluator = Evaluator(
 env=make_eval_env,
 policy_factory=make_eval_policy,
 weight_sync_schemes={
 "policy": MultiProcessWeightSyncScheme(),
 "env.transform[0]": MultiProcessWeightSyncScheme(),
 },
 max_steps=1000,
)

# Sync both policy and env transform via schemes
evaluator.trigger_eval(
 weights_dict={
 "policy": train_policy,
 "env.transform[0]": train_env.transform[0],
 },
 step=current_step,
)

result = evaluator.poll()
```

### VecNormV2 Running Stats

When using the `"tensordict"` strategy (default), the
[`WeightStrategy`](generated/torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) captures `get_extra_state()` alongside
parameters from `TensorDict.from_module()`. This means running
statistics from [`VecNormV2`](generated/torchrl.envs.transforms.VecNormV2.html#torchrl.envs.transforms.VecNormV2) (running
mean, variance, count) are automatically synchronized. No special
handling is needed:

```
from torchrl.weight_update import MultiProcessWeightSyncScheme

# VecNormV2 running stats are automatically captured by the
# tensordict strategy via get_extra_state() / set_extra_state()
evaluator = Evaluator(
 env=make_eval_env, # eval env with VecNormV2 transform
 policy_factory=make_eval_policy,
 weight_sync_schemes={
 "policy": MultiProcessWeightSyncScheme(strategy="tensordict"),
 "env.transform[0]": MultiProcessWeightSyncScheme(strategy="tensordict"),
 },
 max_steps=1000,
)

evaluator.trigger_eval(
 weights_dict={
 "policy": train_policy,
 "env.transform[0]": train_env.transform[0], # VecNormV2
 },
 step=step,
)
```

Important

The evaluator **automatically freezes** all
[`VecNormV2`](generated/torchrl.envs.transforms.VecNormV2.html#torchrl.envs.transforms.VecNormV2) (and
[`VecNorm`](generated/torchrl.envs.transforms.VecNorm.html#torchrl.envs.transforms.VecNorm)) transforms in the eval
environment -- whether the env is passed directly or created from a
factory. This means the eval environment uses the training
statistics as-is and does not update them with eval data.

You do **not** need to call `.freeze()` or use `frozen_copy()`
in your eval env factory -- the evaluator handles this.

For **regular collectors** (not the evaluator), workers that
receive VecNormV2 stats via `weight_sync_schemes` should be
frozen explicitly in the env factory:

```
def make_worker_env():
 env = make_base_env()
 for t in env.transform:
 if hasattr(t, "freeze"):
 t.freeze()
 return env
```

## Transports

Transports handle the low-level communication between sender and receiver. Each scheme creates
appropriate transport instances for its workers.

### Transport Interface

All transports implement the `TransportBackend` protocol with a stateless design. The key methods
accept `weights`, `model`, and `strategy` as keyword arguments rather than storing them as
instance attributes:

```
# Transport methods accept model/weights/strategy as kwargs
transport.receive_weights(
 timeout=None, # Optional timeout in seconds (None = blocking)
 weights=buffer, # Pre-allocated weight buffer
 model=policy, # Model to apply weights to
 strategy=strategy, # WeightStrategy for weight application
)

transport.setup_connection_and_weights_on_receiver(
 worker_idx=0,
 weights=buffer,
 model=policy,
 strategy=strategy,
)
```

### Timeout Support

Transports support timeout for non-blocking weight reception:

| Transport | Timeout Support | Notes |
| --- | --- | --- |
| `MPTransport` | ✅ Yes | Uses `queue.get(timeout=...)` |
| `RPCTransport` | ✅ Yes | Uses `irecv(return_premature=True)` with polling |
| `RayTransport` | ✅ Yes | Uses `irecv(return_premature=True)` with polling |
| `DistributedTransport` | ✅ Yes | Uses `irecv(return_premature=True)` with polling |
| `SharedMemTransport` | N/A | Shared memory is instant (no waiting) |

When `timeout=None` (default), the receive operation blocks until weights arrive.
When a timeout is specified, the method returns `None` if the timeout expires before
weights are received.

### Available Transports

| [`TransportBackend`](generated/torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)(*args, **kwargs) | Core data-plane interface for weight communication mechanisms. |
| --- | --- |
| [`InitialSyncTransport`](generated/torchrl.weight_update.InitialSyncTransport.html#torchrl.weight_update.InitialSyncTransport)(*args, **kwargs) | Optional initial-connection capability for weight transports. |
| [`MPTransport`](generated/torchrl.weight_update.MPTransport.html#torchrl.weight_update.MPTransport)(weight_queue[, ack_queue, timeout]) | Multiprocessing transport using queues. |
| [`SharedMemTransport`](generated/torchrl.weight_update.SharedMemTransport.html#torchrl.weight_update.SharedMemTransport)() | Shared memory transport for in-place weight updates. |
| [`RayTransport`](generated/torchrl.weight_update.RayTransport.html#torchrl.weight_update.RayTransport)(*[, remote_actor, worker_idx, ...]) | Ray transport for communicating with a single Ray actor. |
| [`RPCTransport`](generated/torchrl.weight_update.RPCTransport.html#torchrl.weight_update.RPCTransport)([collector_info, ...]) | RPC transport for communicating with a single RPC remote collector. |
| [`DistributedTransport`](generated/torchrl.weight_update.DistributedTransport.html#torchrl.weight_update.DistributedTransport)(*, weights_buffer[, ...]) | torch.distributed transport for communicating with a single distributed worker. |

## Schemes

Schemes orchestrate the weight synchronization lifecycle, managing initialization, connection setup,
and ongoing weight transfers.

Schemes can be selected from the common backend vocabulary while retaining all
constructor options of the concrete scheme:

```
from torchrl.weight_update import WeightSyncScheme

scheme = WeightSyncScheme.from_backend("shared", sync=False)
```

The mappings are `none`/`direct` (no synchronization), `shared`/`thread`,
`process`/`multiprocessing`, `distributed`, `rpc`, and `ray`.

| [`WeightSyncScheme`](generated/torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)([strategy]) | Configuration for how to synchronize ONE model across workers. |
| --- | --- |
| [`WeightStrategy`](generated/torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)([extract_as]) | Unified strategy for weight transmission. |
| [`MultiProcessWeightSyncScheme`](generated/torchrl.weight_update.MultiProcessWeightSyncScheme.html#torchrl.weight_update.MultiProcessWeightSyncScheme)([strategy, sync]) | Weight synchronization for multiprocess operations using queues. |
| [`SharedMemWeightSyncScheme`](generated/torchrl.weight_update.SharedMemWeightSyncScheme.html#torchrl.weight_update.SharedMemWeightSyncScheme)([strategy, sync, ...]) | Weight synchronization using shared memory. |
| [`NoWeightSyncScheme`](generated/torchrl.weight_update.NoWeightSyncScheme.html#torchrl.weight_update.NoWeightSyncScheme)([strategy]) | No-op weight synchronization scheme. |
| [`RayWeightSyncScheme`](generated/torchrl.weight_update.RayWeightSyncScheme.html#torchrl.weight_update.RayWeightSyncScheme)([strategy, backend]) | Weight synchronization for Ray distributed computing. |
| [`RayModuleTransformScheme`](generated/torchrl.weight_update.RayModuleTransformScheme.html#torchrl.weight_update.RayModuleTransformScheme)([strategy, backend]) | Weight synchronization for RayModuleTransform. |
| [`RPCWeightSyncScheme`](generated/torchrl.weight_update.RPCWeightSyncScheme.html#torchrl.weight_update.RPCWeightSyncScheme)([strategy]) | Weight synchronization for torch.distributed.rpc. |
| [`DistributedWeightSyncScheme`](generated/torchrl.weight_update.DistributedWeightSyncScheme.html#torchrl.weight_update.DistributedWeightSyncScheme)([backend, sync, ...]) | Weight synchronization for torch.distributed. |

## Legacy: Weight Updaters

Warning

The WeightUpdater API is deprecated as of the 0.11 release.
The Weight Sync Schemes API provides more flexibility and better compatibility with heavy
weight transfers (e.g., LLMs) and should be preferred for all new code.

| [`WeightUpdaterBase`](generated/torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)() | A base class for updating remote policy weights on inference workers. |
| --- | --- |
| [`VanillaWeightUpdater`](generated/torchrl.collectors.VanillaWeightUpdater.html#torchrl.collectors.VanillaWeightUpdater)(*[, weight_getter]) | A simple implementation of [`WeightUpdaterBase`](generated/torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) for updating local policy weights. |
| [`MultiProcessedWeightUpdater`](generated/torchrl.collectors.MultiProcessedWeightUpdater.html#torchrl.collectors.MultiProcessedWeightUpdater)(*, ...) | A remote weight updater for synchronizing policy weights across multiple processes or devices. |
| [`RayWeightUpdater`](generated/torchrl.collectors.RayWeightUpdater.html#torchrl.collectors.RayWeightUpdater)(policy_weights, ...[, ...]) | A remote weight updater for synchronizing policy weights across remote workers using Ray. |

| [`RPCWeightUpdater`](generated/torchrl.collectors.distributed.RPCWeightUpdater.html#torchrl.collectors.distributed.RPCWeightUpdater)(collector_infos, ...) | A remote weight updater for synchronizing policy weights across remote workers using RPC. |
| --- | --- |
| [`DistributedWeightUpdater`](generated/torchrl.collectors.distributed.DistributedWeightUpdater.html#torchrl.collectors.distributed.DistributedWeightUpdater)(store, ...) | A remote weight updater for synchronizing policy weights across distributed workers. |