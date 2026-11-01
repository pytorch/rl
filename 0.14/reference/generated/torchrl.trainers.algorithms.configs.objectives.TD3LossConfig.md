# torchrl.trainers.algorithms.configs.objectives.TD3LossConfig

*class*torchrl.trainers.algorithms.configs.objectives.TD3LossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *qvalue_network: Any = None*, *action_spec: Any = None*, *bounds: tuple[float] | None = None*, *num_qvalue_nets: int = 2*, *policy_noise: float = 0.2*, *noise_clip: float = 0.5*, *loss_function: str = 'smooth_l1'*, *delay_actor: bool = True*, *delay_qvalue: bool = True*, *gamma: float | None = None*, *priority_key: str | None = None*, *separate_losses: bool = False*, *reduction: str | None = None*, *deactivate_vmap: bool = False*, *use_prioritized_weights: str | bool = 'auto'*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_td3_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#TD3LossConfig)

Hydra configuration for [`TD3Loss`](torchrl.objectives.TD3Loss.html#torchrl.objectives.TD3Loss).

Every kwarg accepted by `TD3Loss.__init__` is exposed as a field here.