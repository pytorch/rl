# torchrl.trainers.algorithms.configs.objectives.CQLLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.CQLLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *qvalue_network: Any = None*, *loss_function: str = 'smooth_l1'*, *alpha_init: float = 1.0*, *min_alpha: float | None = None*, *max_alpha: float | None = None*, *action_spec: Any = None*, *fixed_alpha: bool = False*, *target_entropy: str | float = 'auto'*, *delay_actor: bool = False*, *delay_qvalue: bool = True*, *gamma: float | None = None*, *temperature: float = 1.0*, *min_q_weight: float = 1.0*, *max_q_backup: bool = False*, *deterministic_backup: bool = True*, *num_random: int = 10*, *with_lagrange: bool = False*, *lagrange_thresh: float = 0.0*, *reduction: str | None = None*, *deactivate_vmap: bool = False*, *scalar_output_mode: str | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_cql_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#CQLLossConfig)

Hydra configuration for [`CQLLoss`](torchrl.objectives.CQLLoss.html#torchrl.objectives.CQLLoss).

Every kwarg accepted by `CQLLoss.__init__` is exposed as a field here.