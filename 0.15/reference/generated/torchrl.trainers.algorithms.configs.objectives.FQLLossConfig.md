# torchrl.trainers.algorithms.configs.objectives.FQLLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.FQLLossConfig(*_partial_: bool = False*, *flow_policy: Any = None*, *actor_network: Any = None*, *qvalue_network: Any = None*, *num_qvalue_nets: int = 2*, *alpha: float = 10.0*, *q_aggregation: str = 'mean'*, *normalize_q_loss: bool = False*, *reduction: str = 'mean'*, *gamma: float = 0.99*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives.make_fql_loss'*, *_convert_: str = 'object'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#FQLLossConfig)

Hydra configuration for [`FQLLoss`](torchrl.objectives.FQLLoss.html#torchrl.objectives.FQLLoss).