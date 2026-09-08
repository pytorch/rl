# DreamerV3LossConfig

*class*torchrl.trainers.algorithms.configs.DreamerV3LossConfig(*_partial_: bool = False*, *model_loss: Any = None*, *actor_loss: Any = None*, *value_loss: Any = None*, *replay_value_loss_weight: float = 0.3*, *continuation_horizon: float = 333.0*, *lmbda: float = 0.95*, *_target_: str = 'torchrl.objectives.DreamerV3Loss'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/objectives.html#DreamerV3LossConfig)

Hydra configuration for [`DreamerV3Loss`](torchrl.objectives.DreamerV3Loss.html#torchrl.objectives.DreamerV3Loss).

Examples

With the component losses from the `DreamerV3Loss` example:

```
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3LossConfig
>>> configured_loss = instantiate(
... DreamerV3LossConfig(), model_loss=model_loss,
... actor_loss=actor_loss, value_loss=value_loss,
... )
>>> losses = configured_loss(sample)
>>> assert not losses["replay_context", "state"].requires_grad
```