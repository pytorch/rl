# torchrl.trainers.algorithms.configs.objectives.DDPGLossConfig

*class*torchrl.trainers.algorithms.configs.objectives.DDPGLossConfig(*_partial_: bool = False*, *actor_network: Any = None*, *value_network: Any = None*, *loss_function: str = 'l2'*, *delay_actor: bool = False*, *delay_value: bool = True*, *gamma: float | None = None*, *separate_losses: bool = False*, *reduction: str | None = None*, *use_prioritized_weights: str | bool = 'auto'*, *_target_: str = 'torchrl.trainers.algorithms.configs.objectives._make_ddpg_loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#DDPGLossConfig)

Hydra configuration for [`DDPGLoss`](torchrl.objectives.DDPGLoss.html#torchrl.objectives.DDPGLoss).

Every kwarg accepted by `DDPGLoss.__init__` is exposed as a field here.