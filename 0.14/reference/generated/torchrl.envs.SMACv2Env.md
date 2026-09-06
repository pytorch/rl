# SMACv2Env

torchrl.envs.SMACv2Env(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/smacv2.html#SMACv2Env)

SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

To install the environment follow the following [guide](https://github.com/oxwhirl/smacv2#getting-started).

Examples

```
>>> from torchrl.envs.libs.smacv2 import SMACv2Env
>>> print(SMACv2Env.available_envs)
['10gen_terran', '10gen_zerg', '10gen_protoss', '3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m',
 '27m_vs_30m', 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '3s_vs_4z', '3s_vs_5z', '1c3s5z',
 '2m_vs_1z', 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane', '2c_vs_64zg']
>>> # You can use old SMAC maps
>>> env = SMACv2Env(map_name="MMM2")
>>> print(env.rollout(5)
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.int64, is_shared=False),
 action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5, 10]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 info: TensorDict(
 fields={
 battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
 dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
 dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
 episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 next: TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5, 10]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 info: TensorDict(
 fields={
 battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
 dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
 dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
 episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False)
>>> # Or the new features for procedural generation
>>> distribution_config = {
... "n_units": 5,
... "n_enemies": 6,
... "team_gen": {
... "dist_type": "weighted_teams",
... "unit_types": ["marine", "marauder", "medivac"],
... "exception_unit_types": ["medivac"],
... "weights": [0.5, 0.2, 0.3],
... "observe": True,
... },
... "start_positions": {
... "dist_type": "surrounded_and_reflect",
... "p": 0.5,
... "n_enemies": 5,
... "map_x": 32,
... "map_y": 32,
... },
... }
>>> env = SMACv2Env(
... map_name="10gen_terran",
... capability_config=distribution_config,
... categorical_actions=False,
... )
>>> print(env.rollout(4))
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.int64, is_shared=False),
 action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 5]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 info: TensorDict(
 fields={
 battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
 dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
 episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False),
 next: TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 5]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 info: TensorDict(
 fields={
 battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
 dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
 dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
 episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False),
 reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False),
 state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False)
```