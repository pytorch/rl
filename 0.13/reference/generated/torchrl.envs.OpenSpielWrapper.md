# OpenSpielWrapper

torchrl.envs.OpenSpielWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/openspiel.html#OpenSpielWrapper)

Google DeepMind OpenSpiel environment wrapper.

GitHub: [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel)

Documentation: [https://openspiel.readthedocs.io/en/latest/index.html](https://openspiel.readthedocs.io/en/latest/index.html)

Parameters:

**env** (*pyspiel.State*) - the game to wrap.

Keyword Arguments:

- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.
- **group_map** ([*MarlGroupMapType*](torchrl.envs.MarlGroupMapType.html#torchrl.envs.MarlGroupMapType)*or**Dict**[**str**,**List**[**str**]**]**]**,**optional*) - how to
group agents in tensordicts for input/output. See
`MarlGroupMapType` for more info.
Defaults to
`ALL_IN_ONE_GROUP`.
- **categorical_actions** (*bool**,**optional*) - if `True`, categorical specs
will be converted to the TorchRL equivalent
([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)), otherwise a one-hot encoding
will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)). Defaults to `False`.
- **return_state** (*bool**,**optional*) - if `True`, "state" is included in the
output of [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) and [`step()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.step). The state can be given
to [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) to reset to that state, rather than resetting to
the initial state.
Defaults to `False`.

Variables:

**available_envs** - environments available to build

Examples

```
>>> import pyspiel
>>> from torchrl.envs import OpenSpielWrapper
>>> from tensordict import TensorDict
>>> base_env = pyspiel.load_game('chess').new_initial_state()
>>> env = OpenSpielWrapper(base_env, return_state=True)
>>> td = env.reset()
>>> td = env.step(env.full_action_spec.rand())
>>> print(td)
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 4672]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 agents: TensorDict(
 fields={
 observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False),
 current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
 3009
 , batch_size=torch.Size([]), device=None),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(env.available_envs)
['2048', 'add_noise', 'amazons', 'backgammon', ...]
```

[`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) can restore a specific state, rather than the initial
state, as long as `return_state=True`.

```
>>> import pyspiel
>>> from torchrl.envs import OpenSpielWrapper
>>> from tensordict import TensorDict
>>> base_env = pyspiel.load_game('chess').new_initial_state()
>>> env = OpenSpielWrapper(base_env, return_state=True)
>>> td = env.reset()
>>> td = env.step(env.full_action_spec.rand())
>>> td_restore = td["next"]
>>> td = env.step(env.full_action_spec.rand())
>>> # Current state is not equal `td_restore`
>>> (td["next"] == td_restore).all()
False
>>> td = env.reset(td_restore)
>>> # After resetting, now the current state is equal to `td_restore`
>>> (td == td_restore).all()
True
```