# ProcgenWrapper

torchrl.envs.ProcgenWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/procgen.html#ProcgenWrapper)

OpenAI Procgen environment wrapper.

Wraps an existing `procgen.ProcgenEnv` instance and exposes it
under the TorchRL environment API.

This wrapper is responsible for:
- Converting Procgen observations (`{"rgb": np.ndarray}`) to Torch tensors
- Handling vectorized Procgen semantics
- Producing TorchRL-compliant `TensorDict` outputs

Parameters:

**env** (*procgen.ProcgenEnv*) - an already constructed Procgen environment.

Keyword Arguments:

- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*|**str**,**optional*) - device on which tensors are placed.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - expected batch size.
- **allow_done_after_reset** (*bool**,**optional*) - tolerate done right after reset.

Variables:

**available_envs** (*List**[**str**]*) - list of Procgen environment ids.

Examples

```
>>> import procgen
>>> from torchrl.envs.libs.procgen import ProcgenWrapper
>>> env = procgen.ProcgenEnv(4, "coinrun")
>>> env = ProcgenWrapper(env=env)
>>> td = env.reset()
>>> print(td)
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 3, 64, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False
)
>>> print(td["observation"].shape)
torch.Size([4, 3, 64, 64])
```