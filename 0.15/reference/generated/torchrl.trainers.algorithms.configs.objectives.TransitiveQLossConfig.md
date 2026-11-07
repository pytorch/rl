# torchrl.trainers.algorithms.configs.objectives.TransitiveQLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.TransitiveQLossConfig(*_partial_: bool = False*, *actor_network: Any = '???'*, *qvalue_network: Any = '???'*, *oracle_qvalue_network: Any = None*, *num_qvalue_nets: int = 2*, *discount: float = 0.99*, *expectile: float = 0.7*, *distance_weight: float = 0.0*, *alpha: float = 1.0*, *actor_action: str = 'mode'*, *_target_: str = 'torchrl.objectives.TransitiveQLoss'*, *_convert_: str = 'object'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#TransitiveQLossConfig)

Hydra configuration for [`TransitiveQLoss`](torchrl.objectives.TransitiveQLoss.html#torchrl.objectives.TransitiveQLoss).

The actor, critic and oracle critic networks are supplied as instantiated
modules (or lists of modules for ensembles).