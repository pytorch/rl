# ProcessInferenceServer

*class*torchrl.modules.inference_server.ProcessInferenceServer(***, *policy_factory: Callable[[], [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]*, *transport: [InferenceTransport](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)*, *max_batch_size: int | None = None*, *min_batch_size: int | None = None*, *timeout: float | None = None*, *collate_fn: Callable | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *policy_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *output_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *collect_stats: bool | None = None*, *stats_window_size: int | None = None*, *weight_sync=None*, *weight_sync_model_id: str = 'policy'*, *server_config: [InferenceServerConfig](torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig) | None = None*, *device_config: [InferenceDeviceConfig](torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig) | None = None*, *policy_version: int = 0*, *policy_version_key: NestedKey | None = 'policy_version'*, *mp_context: str | BaseContext | None = None*, *startup_timeout: float = 300.0*)[[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer)

Dedicated-process wrapper around [`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer).

This server is intended for actor/env workers that communicate through a
queue-based transport such as
[`MPTransport`](torchrl.modules.inference_server.MPTransport.html#torchrl.modules.inference_server.MPTransport). The restricted
client returned by `client()` is created before the server process is
spawned so its response queue is inherited safely.

Parameters:

- **policy_factory** (*Callable**[**[**]**,**nn.Module**]*) - picklable factory that creates
the policy inside the server process.
- **transport** ([*InferenceTransport*](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)) - transport shared with actor clients.

Keyword Arguments:

- **max_batch_size** (*int**,**optional*) - maximum requests per forward pass.
- **min_batch_size** (*int**,**optional*) - minimum requests to accumulate before
dispatching a partial batch.
- **timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - wait timeout in seconds.
- **collate_fn** (*Callable**,**optional*) - collate function for requests.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - alias for `policy_device`.
- **policy_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - policy execution device.
- **output_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - actor response device.
- **collect_stats** (*bool**,**optional*) - forwarded to [`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer).
- **stats_window_size** (*int**,**optional*) - forwarded to [`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer).
- **weight_sync** - optional weight synchronization scheme.
- **weight_sync_model_id** (*str**,**optional*) - model id for weight sync.
- **server_config** ([*InferenceServerConfig*](torchrl.modules.inference_server.InferenceServerConfig.html#torchrl.modules.inference_server.InferenceServerConfig)*,**optional*) - structured server
configuration. Mutually exclusive with the `max_batch_size`,
`min_batch_size`, `timeout`, `collect_stats`, and
`stats_window_size` keyword arguments.
- **device_config** ([*InferenceDeviceConfig*](torchrl.modules.inference_server.InferenceDeviceConfig.html#torchrl.modules.inference_server.InferenceDeviceConfig)*,**optional*) - structured device
placement configuration. Mutually exclusive with `device`,
`policy_device`, and `output_device`. Same field subset as
[`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer): `storing_device` is rejected.
- **policy_version** (*int**,**optional*) - initial behavior-policy version
attached to inference outputs. Defaults to `0`.
- **policy_version_key** (*NestedKey**or**None**,**optional*) - TensorDict key used
for behavior-policy version annotations. `None` disables
annotations. Defaults to `"policy_version"`.
- **mp_context** - multiprocessing context or start-method name. Defaults to
`"spawn"`.
- **startup_timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - seconds `start()` waits for the
child process to build the policy and report readiness. Increase
this when the policy factory loads a large checkpoint. Defaults to
`300.0`.

Examples

```
>>> import multiprocessing as mp
>>> import torch.nn as nn
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import MPTransport
>>> def make_policy():
... return TensorDictModule(
... nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> ctx = mp.get_context("spawn")
>>> transport = MPTransport(ctx=ctx)
>>> server = ProcessInferenceServer(
... policy_factory=make_policy,
... transport=transport,
... mp_context=ctx,
... )
>>> server.start()
>>> client = server.client()
>>> server.shutdown()
```

client() → Any[[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.client)

Return a restricted inference client from the owned transport.

clients(*num_clients: int*) → list[Any][[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.clients)

Return one independently routed client per concurrent consumer.

health(***, *timeout: float = 5.0*) → dict[str, int | bool | None][[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.health)

Return a lightweight child-process health snapshot.

Never raises on a dead or unresponsive child; degraded fields are
reported in the returned dictionary instead (`process_alive` /
`control_error`), so this is safe to call from monitoring loops.

Parameters:

**timeout** (*float**,**optional*) - seconds to wait for the child's
answer. Defaults to `5.0`.

*property*is_alive*: bool*

Whether the child process is alive.

*property*policy_version*: int*

The live behavior-policy version of the child server.

*property*service_backend*: str*

Execution backend that owns the policy.

shutdown(*timeout: float | None = 5.0*) → None[[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.shutdown)

Signal the child process to stop and wait for it to exit.

start() → ProcessInferenceServer[[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.start)

Start the child process and wait until the policy is initialized.

stats(***, *reset: bool = False*, *timeout: float = 5.0*) → dict[str, float | int][[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.stats)

Return process-server stats from the child process.

This is a blocking control-plane round trip: it can take up to
`timeout` seconds and raises `TimeoutError` when the child
does not answer in time, or `RuntimeError` when the child is
not running.

Parameters:

- **reset** (*bool**,**optional*) - if `True`, reset counters in the child
process after taking the snapshot.
- **timeout** (*float**,**optional*) - seconds to wait for the child's
answer. Defaults to `5.0`.

*property*transport_kind*: str*

Physical transport used for inference payloads.

update_model_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *mark_weight_update: bool = True*, *timeout: float = 300.0*) → dict[str, bool][[source]](../../_modules/torchrl/modules/inference_server/_server.html#ProcessInferenceServer.update_model_weights)

Apply TensorDict weights to the model hosted by the child process.

This is a blocking control-plane round trip; large models can take a
while to transfer and apply, hence the generous default timeout.

Parameters:

- **weights** (*TensorDictBase*) - weights to apply to the child's model.
- **mark_weight_update** (*bool**,**optional*) - whether to bump the child's
behavior-policy version. Defaults to `True`.
- **timeout** (*float**,**optional*) - seconds to wait for the child to apply
the weights. Defaults to `300.0`.