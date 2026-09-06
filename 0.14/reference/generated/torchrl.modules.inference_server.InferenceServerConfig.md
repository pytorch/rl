# InferenceServerConfig

*class*torchrl.modules.inference_server.InferenceServerConfig(*service_backend: Literal['thread', 'process', 'ray'] = 'thread'*, *max_batch_size: int = 64*, *min_batch_size: int = 1*, *timeout: float = 0.01*, *collect_stats: bool = True*, *stats_window_size: int = 1024*, *max_inflight_per_env: int | None = None*)[[source]](../../_modules/torchrl/modules/inference_server/_config.html#InferenceServerConfig)

Server-side execution, batching, timeout, and instrumentation settings.

Parameters:

- **service_backend** (*str**,**optional*) - execution backend for the policy server.
`"thread"` runs the serve loop in a background thread of the
constructing process; `"process"` runs a dedicated server
process (which requires a picklable `policy_factory` and a
multiprocessing-capable transport); `"ray"` runs a dedicated
Ray actor and requires `policy_factory`. Defaults to `"thread"`.
- **max_batch_size** (*int**,**optional*) - maximum number of requests per forward
pass. Defaults to `64`.
- **min_batch_size** (*int**,**optional*) - minimum number of requests to
accumulate after the first request arrives. Defaults to `1`.
- **timeout** (*float**,**optional*) - seconds to wait for more requests before
flushing a partial batch. Defaults to `0.01`.
- **collect_stats** (*bool**,**optional*) - whether to collect lightweight
throughput and latency stats. Defaults to `True`.
- **stats_window_size** (*int**,**optional*) - number of recent timing samples kept
for percentile stats. Defaults to `1024`.
- **max_inflight_per_env** (*int**,**optional*) - maximum unresolved remote-policy
requests each environment coordinator may have inflight (consumed
by [`AsyncBatchedCollector`](torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector) when
building its clients). Defaults to `None` (unbounded), so the
guard never throttles by surprise; set an explicit bound when
backpressure is wanted.

Examples

```
>>> import torch
>>> import torch.nn as nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceServer,
... InferenceServerConfig,
... )
>>> policy = TensorDictModule(
... nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> config = InferenceServerConfig(max_batch_size=8, timeout=0.001)
>>> with InferenceServer(policy, transport="auto", server_config=config) as server:
... client = server.client()
... result = client(TensorDict({"observation": torch.randn(4)}))
>>> result["action"].shape
torch.Size([2])
>>> server.max_batch_size
8
```