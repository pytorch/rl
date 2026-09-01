# MultiThreadedEnvWrapper

torchrl.envs.MultiThreadedEnvWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/envpool.html#MultiThreadedEnvWrapper)

Wrapper for envpool-based multithreaded environments.

GitHub: [sail-sg/envpool](https://github.com/sail-sg/envpool)

Paper: [https://arxiv.org/abs/2206.10558](https://arxiv.org/abs/2206.10558)

EnvPool environments auto-reset internally when episodes end. This wrapper
handles that behavior by caching the auto-reset observations and returning
them appropriately in step_and_maybe_reset.

Parameters:

- **env** (*envpool.python.envpool.EnvPoolMixin*) - the envpool to wrap.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.

Keyword Arguments:

- **disable_env_checker** (*bool**,**optional*) - for gym > 0.24 only. If `True` (default
for these versions), the environment checker won't be run.
- **frame_skip** (*int**,**optional*) - if provided, indicates for how many steps the
same action is to be repeated. The observation returned will be the
last observation of the sequence, whereas the reward will be the sum
of rewards across steps.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `torch.device("cpu")`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**batch_size** - The number of envs run simultaneously.

Examples

```
>>> import envpool
>>> from torchrl.envs import MultiThreadedEnvWrapper
>>> env_base = envpool.make(
... task_id="Pong-v5", env_type="gym", num_envs=4, gym_reset_return_info=True
... )
>>> env = MultiThreadedEnvWrapper(envpool_env)
>>> env.reset()
>>> env.rand_step()
```