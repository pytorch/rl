# torchrl.trainers.algorithms.configs.objectives.DQNLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.DQNLossConfig(*_partial_: bool = False*, *value_network: Any = None*, *loss_function: str = 'l2'*, *delay_value: bool = True*, *double_dqn: bool = False*, *action_space: Any = None*, *gamma: float | None = None*, *priority_key: Any = None*, *reduction: str | None = None*, *use_prioritized_weights: str | bool = 'auto'*, *action_key: Any = None*, *action_value_key: Any = None*, *value_key: Any = None*, *reward_key: Any = None*, *done_key: Any = None*, *terminated_key: Any = None*, *priority_weight_key: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_dqn_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#DQNLossConfig)

Hydra configuration for [`DQNLoss`](torchrl.objectives.DQNLoss.html#torchrl.objectives.DQNLoss).

Every kwarg accepted by `DQNLoss.__init__` is exposed as a field here.