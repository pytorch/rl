# check_marl_grouping

torchrl.envs.check_marl_grouping(*group_map: dict[str, list[str]]*, *agent_names: list[str]*)[[source]](../../_modules/torchrl/envs/utils.html#check_marl_grouping)

Check MARL group map.

Performs checks on the group map of a marl environment to assess its validity.
Raises an error in cas of an invalid group_map.

Parameters:

- **group_map** (*Dict**[**str**,**List**[**str**]**]*) - the group map mapping group names to list of agent names in the group
- **agent_names** (*List**[**str**]*) - a list of all the agent names in the environment4

Examples

```
>>> from torchrl.envs.utils import MarlGroupMapType, check_marl_grouping
>>> agent_names = ["agent_0", "agent_1", "agent_2"]
>>> check_marl_grouping(MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names), agent_names)
```