# exploration_type

torchrl.envs.exploration_type() → [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) | None

Returns the interaction type set by `set_interaction_type`, or `None` outside of it.

When it returns `None`, each `ProbabilisticTensorDictModule`
uses its `default_interaction_type`.