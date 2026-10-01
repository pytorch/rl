# torchrl.trainers.algorithms.configs.objectives.TdMpc2LossConfig

*class*torchrl.trainers.algorithms.configs.objectives.TdMpc2LossConfig(*_partial_: bool = False*, *world_model: Any = '???'*, *policy_prior: Any = '???'*, *q_ensemble: Any = '???'*, *horizon: int = 3*, *discount: float = 0.99*, *rho: float = 0.5*, *consistency_coef: float = 20.0*, *reward_coef: float = 0.1*, *value_coef: float = 0.1*, *entropy_coef: float = 0.0001*, *scale_tau: float = 0.01*, *observation_key: Any = 'observation'*, *action_key: Any = 'action'*, *reward_key: Any = 'reward'*, *terminated_key: Any = 'terminated'*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_tdmpc2_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#TdMpc2LossConfig)

Hydra configuration for [`TdMpc2Loss`](torchrl.objectives.TdMpc2Loss.html#torchrl.objectives.TdMpc2Loss).

This configuration is intended to be instantiated with shared world-model,
policy-prior, and Q-ensemble instances supplied by a trainer factory.