# MultiThreadedEnv

torchrl.envs.MultiThreadedEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/envpool.html#MultiThreadedEnv)

Multithreaded execution of environments based on EnvPool.

GitHub: [sail-sg/envpool](https://github.com/sail-sg/envpool)

Paper: [https://arxiv.org/abs/2206.10558](https://arxiv.org/abs/2206.10558)

An alternative to ParallelEnv based on multithreading. It's faster, as it doesn't require new process spawning, but
less flexible, as it only supports environments implemented in EnvPool library.
Currently, only supports synchronous execution mode, when the batch size is equal to the number of workers, see
[https://envpool.readthedocs.io/en/latest/content/python_interface.html#batch-size](https://envpool.readthedocs.io/en/latest/content/python_interface.html#batch-size).

Parameters:

- **num_workers** (*int*) - The number of envs to run simultaneously. Will be
identical to the content of ~.batch_size.
- **env_name** (*str*) - name of the environment to build.

Keyword Arguments:

- **create_env_kwargs** (*Dict**[**str**,**Any**]**,**optional*) - kwargs to be passed to envpool
environment constructor.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.
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

Examples

```
>>> env = MultiThreadedEnv(num_workers=3, env_name="Pendulum-v1")
>>> env.reset()
>>> env.rand_step()
>>> env.rollout(5)
>>> env.close()
```