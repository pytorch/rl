# torchrl.trainers.algorithms.configs.modules.DreamerV3DiscreteActorConfig

*class*torchrl.trainers.algorithms.configs.modules.DreamerV3DiscreteActorConfig(*_partial_: bool = False*, *in_features: int = '???'*, *out_features: int = '???'*, *depth: int = 3*, *num_cells: int = 1024*, *norm_eps: float = 0.0001*, *unimix: float = 0.01*, *in_keys: Any = None*, *action_key: Any = 'action'*, *logits_key: Any = 'logits'*, *log_prob_key: Any = 'action_log_prob'*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_dreamer_v3_discrete_actor'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#DreamerV3DiscreteActorConfig)

Hydra configuration for `DreamerV3DiscreteActor`.

Examples

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from tensordict import TensorDict
>>> from torchrl.trainers.algorithms.configs import DreamerV3DiscreteActorConfig
>>> actor = instantiate(DreamerV3DiscreteActorConfig(in_features=12, out_features=3))
>>> data = TensorDict({"state": torch.randn(4, 8), "belief": torch.randn(4, 4)}, [4])
>>> actor(data)["action"].shape
torch.Size([4, 3])
```