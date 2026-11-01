# RoboHiveEnv

torchrl.envs.RoboHiveEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/robohive.html#RoboHiveEnv)

A wrapper for RoboHive gym environments.

RoboHive is a collection of environments/tasks simulated with the MuJoCo physics engine exposed using the OpenAI-Gym API.

Github: [vikashplus/robohive](https://github.com/vikashplus/robohive/)

Doc: [vikashplus/robohive](https://github.com/vikashplus/robohive/wiki)

Paper: [https://arxiv.org/abs/2310.06828](https://arxiv.org/abs/2310.06828)

Warning

RoboHive requires gym 0.13.

Parameters:

- **env_name** (*str*) - the environment name to build. Must be one of `available_envs`
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
- **from_depths** (*bool**,**optional*) - if `True`, an attempt to return the depth
observations from the env will be performed. By default, these observations
will be written under the `"depths"` entry. Requires `from_pixels` to be `True`.
Defaults to `False`.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - Only `torch.Size([])` will work with
`RoboHiveEnv` since vectorized environments are not supported within the
class. To execute more than one environment at a time, see [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv).
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**available_envs** (*list*) - a list of available envs to build.

Examples

```
>>> from torchrl.envs import RoboHiveEnv
>>> env = RoboHiveEnv(RoboHiveEnv.available_envs[0])
>>> env.rollout(3)
```