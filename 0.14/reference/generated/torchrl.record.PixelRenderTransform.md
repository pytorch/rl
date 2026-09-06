# PixelRenderTransform

torchrl.record.PixelRenderTransform(*out_keys: list[NestedKey] = None*, *preproc: Callable[[np.ndarray | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], np.ndarray | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] = None*, *as_non_tensor: bool | None = None*, *render_method: str = 'render'*, *pass_tensordict: bool = False*, ***kwargs*) → None[[source]](../../_modules/torchrl/record/recorder.html#PixelRenderTransform)

A transform to call render on the parent environment and register the pixel observation in the tensordict.

This transform offers an alternative to the `from_pixels` syntactic sugar when instantiating an environment
that offers rendering is expensive, or when `from_pixels` is not implemented.
It can be used within a single environment or over batched environments alike.

Parameters:

- **out_keys** (*List**[**NestedKey**] or**Nested*) - List of keys where to register the pixel observations.
- **preproc** (*Callable**,**optional*) - a preproc function. Can be used to reshape the observation, or apply
any other transformation that makes it possible to register it in the output data.
- **as_non_tensor** (*bool**,**optional*) - if `True`, the data will be written as a [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)
thereby relaxing the shape requirements. If not provided, it will be inferred automatically from the
input data type and shape.
- **render_method** (*str**,**optional*) - the name of the render method. Defaults to `"render"`.
- **pass_tensordict** (*bool**,**optional*) - if `True`, the input tensordict will be passed to the
render method. This enables rendering for stateless environments. Defaults to `False`.
- ****kwargs** - additional keyword arguments to pass to the render function (e.g. `mode="rgb_array"`).

Examples

```
>>> from torchrl.envs import GymEnv, check_env_specs, ParallelEnv, EnvCreator
>>> from torchrl.record.loggers import CSVLogger
>>> from torchrl.record.recorder import PixelRenderTransform, VideoRecorder
>>>
>>> def make_env():
>>> env = GymEnv("CartPole-v1", render_mode="rgb_array")
>>> env = env.append_transform(PixelRenderTransform())
>>> return env
>>>
>>> if __name__ == "__main__":
... logger = CSVLogger("dummy", video_format="mp4")
...
... env = ParallelEnv(4, EnvCreator(make_env))
...
... env = env.append_transform(VideoRecorder(logger=logger, tag="pixels_record"))
... env.rollout(3)
...
... check_env_specs(env)
...
... r = env.rollout(30)
... print(env)
... env.transform.dump()
... env.close()
```

This transform can also be used whenever a batched environment `render()` returns a single image:

Examples

```
>>> from torchrl.envs import check_env_specs
>>> from torchrl.envs.libs.vmas import VmasEnv
>>> from torchrl.record.loggers import CSVLogger
>>> from torchrl.record.recorder import PixelRenderTransform, VideoRecorder
>>>
>>> env = VmasEnv(
... scenario="flocking",
... num_envs=32,
... continuous_actions=True,
... max_steps=200,
... device="cpu",
... seed=None,
... # Scenario kwargs
... n_agents=5,
... )
>>>
>>> logger = CSVLogger("dummy", video_format="mp4")
>>>
>>> env = env.append_transform(PixelRenderTransform(mode="rgb_array", preproc=lambda x: x.copy()))
>>> env = env.append_transform(VideoRecorder(logger=logger, tag="pixels_record"))
>>>
>>> check_env_specs(env)
>>>
>>> r = env.rollout(30)
>>> env.transform[-1].dump()
```

The transform can be disabled using the `switch()` method, which will
turn the rendering on if it's off or off if it's on (an argument can also be passed to control this behavior).
Since transforms are [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) instances, [`apply()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply) can be used to control
this behavior:

```
>>> def switch(module):
... if isinstance(module, PixelRenderTransform):
... module.switch()
>>> env.apply(switch)
```