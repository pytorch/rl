# InferenceClient

*class*torchrl.modules.inference_server.InferenceClient(*transport: [InferenceTransport](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)*)[[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceClient)

Actor-side handle for an [`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer).

Wraps a transport's [`submit()`](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport.submit) so that calling
`client(td)` looks like a regular synchronous policy call, while the
actual computation is batched on the server.

Parameters:

**transport** ([*InferenceTransport*](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)) - the transport shared with the server.

Example

```
>>> client = transport.client()
>>> td_out = client(td_in) # blocking
>>> future = client.submit(td_in) # non-blocking
>>> td_out = future.result()
```

submit(*td: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → Future[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/modules/inference_server/_server.html#InferenceClient.submit)

Submit a request and return a Future immediately.