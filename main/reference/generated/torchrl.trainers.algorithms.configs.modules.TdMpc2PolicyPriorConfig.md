# torchrl.trainers.algorithms.configs.modules.TdMpc2PolicyPriorConfig

*class*torchrl.trainers.algorithms.configs.modules.TdMpc2PolicyPriorConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *latent_dim: int = '???'*, *action_dim: int = '???'*, *mlp_dim: int = 512*, *log_std_min: float = -10.0*, *log_std_max: float = 2.0*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tdmpc2_policy_prior'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TdMpc2PolicyPriorConfig)

Configuration for a TD-MPC2 policy prior.

The policy prior maps a latent state to a sampled action and the
distribution statistics used by the TD-MPC2 objective. By default, the
module reads `"latent"` and writes `"action"`, "mean"`,
``"log_std", `"entropy"`, and `"scaled_entropy"`.

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from tensordict import TensorDict
>>> from torchrl.trainers.algorithms.configs import TdMpc2PolicyPriorConfig
>>> cfg = TdMpc2PolicyPriorConfig(latent_dim=8, action_dim=2, mlp_dim=16)
>>> policy = instantiate(cfg)
>>> td = TensorDict({"latent": torch.randn(3, 8)}, batch_size=[3])
>>> assert policy(td)["action"].shape == (3, 2)
```