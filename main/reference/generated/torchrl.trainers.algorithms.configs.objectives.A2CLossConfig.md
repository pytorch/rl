# torchrl.trainers.algorithms.configs.objectives.A2CLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.A2CLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *critic_network: Any = None*, *entropy_bonus: bool = True*, *samples_mc_entropy: int = 1*, *entropy_coeff: float | None = None*, *critic_coeff: float = 1.0*, *loss_critic_type: str = 'smooth_l1'*, *gamma: float | None = None*, *separate_losses: bool = False*, *advantage_key: Any = None*, *value_target_key: Any = None*, *functional: bool = True*, *actor: Any = None*, *critic: Any = None*, *reduction: str | None = None*, *clip_value: float | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_a2c_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#A2CLossConfig)

Hydra configuration for [`A2CLoss`](torchrl.objectives.A2CLoss.html#torchrl.objectives.A2CLoss).

Every kwarg accepted by `A2CLoss.__init__` is exposed as a field here.
`gamma`, `advantage_key` and `value_target_key` are handled by the
factory (via `make_value_estimator` and `set_keys`) rather than being
forwarded to the constructor, which rejects them.