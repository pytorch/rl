# CrossCriticGroupSpec

*class*torchrl.modules.CrossCriticGroupSpec(*obs_dim: int*, *n_agents: int*, *obs_key: NestedKey*, *value_key: NestedKey*)[[source]](../../_modules/torchrl/modules/models/cross_group_critic.html#CrossCriticGroupSpec)

Specification for one agent group used by [`CrossGroupCritic`](torchrl.modules.CrossGroupCritic.html#torchrl.modules.CrossGroupCritic).

Parameters:

- **obs_dim** (*int*) - dimensionality of each agent's observation vector.
- **n_agents** (*int*) - number of agents in the group.
- **obs_key** (*NestedKey*) - tensordict key holding this group's observations,
e.g. `("soldiers", "observation")`.
- **value_key** (*NestedKey*) - tensordict key where this group's state values
will be written, e.g. `("soldiers", "state_value")`.