# set_exploration_type

torchrl.envs.set_exploration_type(*type: InteractionType | str | None = InteractionType.DETERMINISTIC*) → None

Sets all ProbabilisticTDModules sampling to the desired type.

Parameters:

**type** (*InteractionType**or**str*) - sampling type to use when the policy is being called.