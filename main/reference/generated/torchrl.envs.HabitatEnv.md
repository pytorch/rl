# HabitatEnv

torchrl.envs.HabitatEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/habitat.html#HabitatEnv)

A wrapper for habitat envs.

This class currently serves as placeholder and compatibility security.
It behaves exactly like the GymEnv wrapper.

Doc: [https://aihabitat.org/docs/](https://aihabitat.org/docs/)

GitHub: [facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)

URL: [https://aihabitat.org/habitat3/](https://aihabitat.org/habitat3/)

Paper: [https://ai.meta.com/static-resource/habitat3](https://ai.meta.com/static-resource/habitat3)

Parameters:

- **env_name** (*str*) - The environment to execute.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.

Keyword Arguments:

- **from_pixels** (*bool**,**optional*) - if `True`, an attempt to return the pixel
observations from the env will be performed. By default, these observations
will be written under the `"pixels"` entry.
The method being used varies
depending on the gym version and may involve a `wrappers.pixel_observation.PixelObservationWrapper`.
Defaults to `False`.
- **pixels_only** (*bool**,**optional*) - if `True`, only the pixel observations will
be returned (by default under the `"pixels"` entry in the output tensordict).
If `False`, observations (eg, states) and pixels will be returned
whenever `from_pixels=True`. Defaults to `True`.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device
on which the simulation will occur. When `num_workers > 1`, this can be a
list of devices (one per worker) to distribute environments across multiple
GPUs. Defaults to `torch.device("cuda:0")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
Should match the leading dimensions of all observations, done states,
rewards, actions and infos.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.
- **num_workers** (*int**,**optional*) - if provided and greater than 1, a
[`torchrl.envs.ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) will be instantiated with
`num_workers` copies of `HabitatEnv`. Defaults to `1`.

Variables:

**available_envs** (*List**[**str**]*) - a list of environments to build.

Examples

```
>>> from torchrl.envs import HabitatEnv
>>> env = HabitatEnv("HabitatRenderPick-v0", from_pixels=True)
>>> env.rollout(3)
```