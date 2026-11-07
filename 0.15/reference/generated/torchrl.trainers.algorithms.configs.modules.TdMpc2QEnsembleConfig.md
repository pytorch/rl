# torchrl.trainers.algorithms.configs.modules.TdMpc2QEnsembleConfig

*class*torchrl.trainers.algorithms.configs.modules.TdMpc2QEnsembleConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *latent_dim: int = '???'*, *action_dim: int = '???'*, *mlp_dim: int = 512*, *num_q: int = 5*, *num_bins: int = 101*, *vmin: float = -10.0*, *vmax: float = 10.0*, *dropout: float = 0.01*, *q_value_key: Any = 'q_value'*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tdmpc2_q_ensemble'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TdMpc2QEnsembleConfig)

Configuration for a TD-MPC2 Q-function ensemble.

The resulting
[`TdMpc2QEnsemble`](torchrl.modules.TdMpc2QEnsemble.html#torchrl.modules.TdMpc2QEnsemble) maps a latent
state and action to distributional Q-function logits. Its `forward()`
method writes all ensemble logits, while
[`reduce()`](torchrl.modules.TdMpc2QEnsemble.html#torchrl.modules.TdMpc2QEnsemble.reduce) decodes two randomly
selected Q-functions and writes a reduced value.

Parameters:

- **latent_dim** - Size of the latent state input.
- **action_dim** - Number of action dimensions.
- **mlp_dim** - Width of each hidden layer in every Q-function.
- **num_q** - Number of Q-functions in the ensemble.
- **num_bins** - Number of categorical bins for the Q-function output. Must be
greater than 1.
- **vmin** - Minimum value of the symlog-space categorical support.
- **vmax** - Maximum value of the symlog-space categorical support.
- **dropout** - Dropout probability applied in the Q-functions.
- **q_value_key** - TensorDict key written by the ensemble reduction.
- **device** - Device on which to construct the Q-functions.

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from tensordict import TensorDict
>>> from torchrl.trainers.algorithms.configs import TdMpc2QEnsembleConfig
>>> cfg = TdMpc2QEnsembleConfig(latent_dim=8, action_dim=2, mlp_dim=16)
>>> q_ensemble = instantiate(cfg)
>>> data = TensorDict(
... {"latent": torch.randn(3, 8), "action": torch.randn(3, 2)},
... batch_size=[3],
... )
>>> q_ensemble(data)["q_logits"].shape
torch.Size([3, 5, 101])
```

See also

[`TdMpc2QEnsemble`](torchrl.modules.TdMpc2QEnsemble.html#torchrl.modules.TdMpc2QEnsemble)