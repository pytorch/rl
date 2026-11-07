# torchrl.trainers.algorithms.configs.transforms.ClosedLoopMultiActionConfig

*class*torchrl.trainers.algorithms.configs.transforms.ClosedLoopMultiActionConfig(*controller: Any = '???'*, *steps: int = '???'*, *decision_spec: Any = None*, *reward_aggregation: str = 'sum'*, *exploration_type: [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) = InteractionType.DETERMINISTIC*, *no_grad: bool = True*, *dim: int = 1*, *stack_observations: bool = False*, *_target_: str = 'torchrl.envs.transforms.ClosedLoopMultiAction'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/transforms.html#ClosedLoopMultiActionConfig)

Hydra configuration for [`ClosedLoopMultiAction`](torchrl.envs.transforms.ClosedLoopMultiAction.html#torchrl.envs.transforms.ClosedLoopMultiAction).

Install controller primers first, or override the target with
ClosedLoopMultiAction.from_env and supply the environment to instantiate.