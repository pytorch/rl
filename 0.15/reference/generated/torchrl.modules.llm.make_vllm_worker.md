# make_vllm_worker

*class*torchrl.modules.llm.make_vllm_worker(***, *model_name: str*, *devices: list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int] | None = None*, *num_devices: int | None = None*, *make_ray_worker: bool = True*, *enforce_eager: bool = False*, *enable_fp32_output: bool = False*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_sync.html#make_vllm_worker)

Creates a vLLM inference engine with tensor parallelism support.

Parameters:

- **model_name** (*str*) - The model name to pass to vLLM.LLM.
- **devices** (*list**[*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*|**int**]**,**optional*) - List of devices to use. Exclusive with num_devices.
- **num_devices** (*int**,**optional*) - Number of devices to use. Exclusive with devices.
- **make_ray_worker** (*bool**,**optional*) - Whether to create a Ray actor. Defaults to True.
- **enforce_eager** (*bool**,**optional*) - Whether to enforce eager execution. Defaults to False.
- **enable_fp32_output** (*bool**,**optional*) - Whether to enable FP32 output for the final layer. Defaults to False.
This can help with numerical stability for certain models. Requires model-specific support in
torchrl.modules.llm.backends.vllm._models.
- ****kwargs** - Additional arguments passed to vLLM.LLM.__init__.

Returns:

Either a Ray worker wrapper or a local LLM wrapper, both implementing RLvLLMEngine.

Return type:

RayLLMWorker | LocalLLMWrapper

Example

```
>>> # Create a 2-GPU tensor parallel worker with Ray
>>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", num_devices=2)
>>> # Create a local LLM instance on GPU 1
>>> llm = make_vllm_worker("Qwen/Qwen2.5-3B", devices=[1], make_ray_worker=False)
>>> # Create with FP32 output enabled
>>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", num_devices=2, enable_fp32_output=True)
```