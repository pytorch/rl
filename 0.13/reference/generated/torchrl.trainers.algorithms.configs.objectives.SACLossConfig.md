# torchrl.trainers.algorithms.configs.objectives.SACLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.SACLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *qvalue_network: Any = None*, *value_network: Any = None*, *discrete: bool = False*, *action_space: Any = None*, *num_actions: int | None = None*, *num_qvalue_nets: int = 2*, *loss_function: str = 'smooth_l1'*, *alpha_init: float = 1.0*, *min_alpha: float | None = None*, *max_alpha: float | None = None*, *action_spec: Any = None*, *fixed_alpha: bool = False*, *target_entropy: str | float = 'auto'*, *target_entropy_weight: float = 0.98*, *delay_actor: bool = False*, *delay_qvalue: bool = True*, *delay_value: bool = True*, *gamma: float | None = None*, *priority_key: str | None = None*, *separate_losses: bool = False*, *reduction: str | None = None*, *skip_done_states: bool = False*, *deactivate_vmap: bool = False*, *use_prioritized_weights: str | bool = 'auto'*, *scalar_output_mode: str | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_sac_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#SACLossConfig)

Hydra configuration for [`SACLoss`](torchrl.objectives.SACLoss.html#torchrl.objectives.SACLoss) (and [`DiscreteSACLoss`](torchrl.objectives.DiscreteSACLoss.html#torchrl.objectives.DiscreteSACLoss) when `discrete=True`).

Every kwarg accepted by `SACLoss.__init__` is exposed as a field here. The
`discrete`/`action_space`/`num_actions`/`target_entropy_weight` fields
apply only when the discrete variant is selected.