# InferenceDeviceConfig

*class*torchrl.modules.inference_server.InferenceDeviceConfig(*policy_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *output_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *env_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *storing_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*)[[source]](../../_modules/torchrl/modules/inference_server/_config.html#InferenceDeviceConfig)

Device placement for asynchronous policy-server collection.

This config separates the devices used by the environment, the remote
policy, the actor-side action TensorDict, and the returned collector batch.

All fields accept [`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device), `str`, or `None` and are
normalized to `torch.device | None` at construction time.

Parameters:

- **policy_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device that owns the
policy and receives batched server inputs.
- **output_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device for inference
results returned by the server.
- **env_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device used by env workers
when stepping environments. If `output_device` is omitted, this is
the natural device for returned actions.
- **storing_device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device used for
collected transitions yielded by the collector.

Examples

```
>>> import torch
>>> import torch.nn as nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceDeviceConfig,
... InferenceServer,
... ThreadingTransport,
... )
>>> policy = TensorDictModule(
... nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> transport = ThreadingTransport()
>>> device_config = InferenceDeviceConfig(
... policy_device="cpu", output_device="cpu"
... )
>>> with InferenceServer(policy, transport, device_config=device_config):
... client = transport.client()
... result = client(TensorDict({"observation": torch.randn(4)}))
>>> result["action"].device.type
'cpu'
```

server_output_device() → [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None[[source]](../../_modules/torchrl/modules/inference_server/_config.html#InferenceDeviceConfig.server_output_device)

Return the actor-side device expected from the policy server.