# torchrl.trainers.algorithms.configs.objectives.PPOLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.PPOLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *critic_network: Any = None*, *loss_type: str = 'clip'*, *entropy_bonus: bool = True*, *samples_mc_entropy: int = 1*, *entropy_coeff: float | None = None*, *log_explained_variance: bool = True*, *critic_coeff: float | None = None*, *loss_critic_type: str = 'smooth_l1'*, *normalize_advantage: bool = False*, *normalize_advantage_exclude_dims: tuple = ()*, *gamma: float | None = None*, *separate_losses: bool = False*, *advantage_key: str | None = None*, *value_target_key: str | None = None*, *value_key: str | None = None*, *functional: bool = True*, *actor: Any = None*, *critic: Any = None*, *reduction: str | None = None*, *clip_value: float | None = None*, *clip_epsilon: Any = 0.2*, *dtarg: float = 0.01*, *beta: float = 1.0*, *increment: float = 2.0*, *decrement: float = 0.5*, *samples_mc_kl: int = 1*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_ppo_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#PPOLossConfig)

Hydra configuration for the PPO loss family.

Dispatches between [`ClipPPOLoss`](torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) (`loss_type='clip'`),
[`KLPENPPOLoss`](torchrl.objectives.KLPENPPOLoss.html#torchrl.objectives.KLPENPPOLoss) (`loss_type='kl'`) and
[`PPOLoss`](torchrl.objectives.PPOLoss.html#torchrl.objectives.PPOLoss) (`loss_type='ppo'`). Every kwarg
accepted by any of those classes is exposed here; only the kwargs relevant
to the selected `loss_type` are forwarded.