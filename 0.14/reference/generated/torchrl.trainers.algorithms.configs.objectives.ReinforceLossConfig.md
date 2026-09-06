# torchrl.trainers.algorithms.configs.objectives.ReinforceLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.ReinforceLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *critic_network: Any = None*, *delay_value: bool = False*, *loss_critic_type: str = 'smooth_l1'*, *gamma: float | None = None*, *advantage_key: Any = None*, *value_target_key: Any = None*, *separate_losses: bool = False*, *functional: bool = True*, *actor: Any = None*, *critic: Any = None*, *reduction: str | None = None*, *clip_value: float | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_reinforce_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#ReinforceLossConfig)

Hydra configuration for [`ReinforceLoss`](torchrl.objectives.ReinforceLoss.html#torchrl.objectives.ReinforceLoss).

Every kwarg accepted by `ReinforceLoss.__init__` is exposed as a field
here. `gamma`, `advantage_key` and `value_target_key` are handled by
the factory (via `make_value_estimator` and `set_keys`) rather than
being forwarded to the constructor, which rejects them.