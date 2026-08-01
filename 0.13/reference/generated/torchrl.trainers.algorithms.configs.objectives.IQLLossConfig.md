# torchrl.trainers.algorithms.configs.objectives.IQLLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.IQLLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *qvalue_network: Any = None*, *value_network: Any = None*, *discrete: bool = False*, *action_space: Any = None*, *num_qvalue_nets: int = 2*, *loss_function: str = 'smooth_l1'*, *temperature: float = 1.0*, *expectile: float = 0.5*, *gamma: float | None = None*, *priority_key: str | None = None*, *separate_losses: bool = False*, *reduction: str | None = None*, *deactivate_vmap: bool = False*, *scalar_output_mode: str | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_iql_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#IQLLossConfig)

Hydra configuration for [`IQLLoss`](torchrl.objectives.IQLLoss.html#torchrl.objectives.IQLLoss) (and [`DiscreteIQLLoss`](torchrl.objectives.DiscreteIQLLoss.html#torchrl.objectives.DiscreteIQLLoss) when `discrete=True`).

Every kwarg accepted by `IQLLoss.__init__` is exposed as a field here. The
`discrete`/`action_space` fields apply only when the discrete variant is
selected.