# Inference Server

The inference server provides auto-batching model serving for RL actors.
Multiple actors submit individual TensorDicts; the server transparently
batches them, runs a single model forward pass, and routes results back.

## Core API

| [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer)(model, transport, *[, ...]) | Auto-batching inference server. |
| --- | --- |
| [`InferenceServerConfig`](generated/torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig)([backend, ...]) | Server-side execution, batching, timeout, and instrumentation settings. |
| [`InferenceDeviceConfig`](generated/torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig)([policy_device, ...]) | Device placement for asynchronous policy-server collection. |
| [`ProcessInferenceServer`](generated/torchrl.modules.inference_server.ProcessInferenceServer.html#torchrl.modules.inference_server.ProcessInferenceServer)(*, policy_factory, ...) | Dedicated-process wrapper around [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer). |
| [`InferenceClient`](generated/torchrl.modules.inference_server.InferenceClient.html#torchrl.modules.inference_server.InferenceClient)(transport) | Actor-side handle for an [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer). |
| [`PolicyClientModule`](generated/torchrl.modules.inference_server.PolicyClientModule.html#torchrl.modules.inference_server.PolicyClientModule)(*args, **kwargs) | TensorDict policy wrapper for remote inference-server clients. |
| [`InferenceTransport`](generated/torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)() | Abstract base class for inference server transport backends. |

## Transport Backends

| [`ThreadingTransport`](generated/torchrl.modules.inference_server.ThreadingTransport.html#torchrl.modules.inference_server.ThreadingTransport)() | In-process transport for actors that are threads. |
| --- | --- |
| [`SlotTransport`](generated/torchrl.modules.inference_server.SlotTransport.html#torchrl.modules.inference_server.SlotTransport)(num_slots, *[, preallocate]) | Lock-free, in-process transport using per-env slots. |
| [`MPTransport`](generated/torchrl.modules.inference_server.MPTransport.html#torchrl.modules.inference_server.MPTransport)([ctx, use_manager]) | Cross-process transport using `multiprocessing` queues. |
| [`RayTransport`](generated/torchrl.modules.inference_server.RayTransport.html#torchrl.modules.inference_server.RayTransport)(*[, max_queue_size]) | Transport using Ray queues for distributed inference. |
| [`MonarchTransport`](generated/torchrl.modules.inference_server.MonarchTransport.html#torchrl.modules.inference_server.MonarchTransport)(*[, max_queue_size]) | Transport using Monarch for distributed inference on GPU clusters. |

## Usage

The simplest setup uses [`ThreadingTransport`](generated/torchrl.modules.inference_server.ThreadingTransport.html#torchrl.modules.inference_server.ThreadingTransport) for actors that are
threads in the same process:

```
from tensordict.nn import TensorDictModule
from torchrl.modules.inference_server import (
 InferenceServer,
 ThreadingTransport,
)
import torch.nn as nn
import concurrent.futures

policy = TensorDictModule(
 nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4)),
 in_keys=["observation"],
 out_keys=["action"],
)

transport = ThreadingTransport()
server = InferenceServer(policy, transport, max_batch_size=32)
server.start()
client = transport.client()

# actor threads call client(td) -- batched automatically
with concurrent.futures.ThreadPoolExecutor(16) as pool:
 ...

server.shutdown()
```

### Structured Configuration

Server execution, batching, and device placement are grouped into two
dataclasses instead of loose keyword arguments: [`InferenceServerConfig`](generated/torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig)
collects the execution `backend` (`"thread"` or `"process"`) and the
batching/instrumentation knobs (`max_batch_size`, `min_batch_size`,
`timeout`, `collect_stats`, `stats_window_size`), and
[`InferenceDeviceConfig`](generated/torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig) describes device placement across the
collection pipeline (`policy_device`, `output_device`, `env_device`,
`storing_device`). Both [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) and
[`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector) accept them through the
`server_config` and `device_config` keyword arguments; a config object is
mutually exclusive with the individual keyword arguments it replaces, and the
config objects are the only way to set the per-role devices and the server
backend on the collector. Servers consume only the
`policy_device`/`output_device` fields (`env_device` doubles as an
`output_device` fallback), while `env_device` and `storing_device`
drive the collector-side transfers:

```
from torchrl.collectors import AsyncBatchedCollector
from torchrl.modules.inference_server import (
 InferenceDeviceConfig,
 InferenceServerConfig,
)

collector = AsyncBatchedCollector(
 create_env_fn=[make_env] * 8,
 policy=my_policy,
 frames_per_batch=200,
 server_config=InferenceServerConfig(max_batch_size=8, timeout=0.005),
 device_config=InferenceDeviceConfig(
 policy_device="cuda:0",
 env_device="cpu",
 storing_device="cpu",
 ),
)
```

### Remote policy module

Use [`PolicyClientModule`](generated/torchrl.modules.inference_server.PolicyClientModule.html#torchrl.modules.inference_server.PolicyClientModule) when an actor or collector expects a regular
TensorDict policy but inference should be served by the policy server:

```
remote_policy = PolicyClientModule(
 transport,
 in_keys=["observation"],
 out_keys=["action", "policy_version"],
)

data = remote_policy(data)
```

The server writes `policy_version` by default so asynchronous collectors can
track behavior-policy lag. This is the general *service-stamped metadata*
pattern: any service may stamp its responses with metadata about the state it
served them from, and the data pipeline may enforce freshness constraints on
it. Bounded staleness is enforced by the replay buffer through
[`PolicyAgeFilter`](generated/torchrl.envs.transforms.PolicyAgeFilter.html#torchrl.envs.transforms.PolicyAgeFilter), which drops elements whose
stamped version lags the live version by more than `max_policy_lag` -
either at extension time or dynamically at sampling time.

### Weight Synchronisation

The server integrates with [`WeightSyncScheme`](generated/torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)
to receive updated model weights from a trainer between inference batches:

```
from torchrl.weight_update import SharedMemWeightSyncScheme

weight_sync = SharedMemWeightSyncScheme()
# Initialise on the trainer (sender) side first
weight_sync.init_on_sender(model=training_model, ...)

server = InferenceServer(
 model=inference_model,
 transport=ThreadingTransport(),
 weight_sync=weight_sync,
)
server.start()

# Training loop
for batch in dataloader:
 loss = loss_fn(training_model(batch))
 loss.backward()
 optimizer.step()
 weight_sync.send(model=training_model) # pushed to server
```

### Integration with Collectors

The easiest way to use the inference server with RL data collection is
through [`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector), which
creates the server, transport, and env pool automatically:

```
from torchrl.collectors import AsyncBatchedCollector
from torchrl.envs import GymEnv

collector = AsyncBatchedCollector(
 create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
 policy=my_policy,
 frames_per_batch=200,
 total_frames=10_000,
 max_batch_size=8,
)

for data in collector:
 # train on data ...
 pass

collector.shutdown()
```