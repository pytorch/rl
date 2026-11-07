# torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig

*class*torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *network: [MLPConfig](torchrl.trainers.algorithms.configs.modules.MLPConfig.html#torchrl.trainers.algorithms.configs.modules.MLPConfig) = '???'*, *eval_mode: bool = False*, *extract_normal_params: bool = True*, *scale_mapping: str = 'biased_softplus_1.0'*, *scale_lb: float = 0.0001*, *low: Any = None*, *high: Any = None*, *tanh_loc: bool = False*, *param_keys: Any = None*, *exploration_type: Any = 'RANDOM'*, *return_log_prob: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tanh_normal_model'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TanhNormalModelConfig)

A class to configure a TanhNormal model.

Example

```
>>> cfg = TanhNormalModelConfig(network=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32))
>>> net = instantiate(cfg)
>>> y = net(torch.randn(1, 10))
>>> assert y.shape == (1, 5)
```

Parameters:

- **low** - lower bound of the action support handed to
[`TanhNormal`](torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal) (a scalar or a per-dimension
sequence). Defaults to `None`, i.e. the distribution default of `-1`.
- **high** - upper bound of the action support. Defaults to `None`, i.e. `1`.
- **tanh_loc** - if `True`, the location is squashed to `[-upscale, upscale]`
before the tanh transform, which keeps the log-probability of actions
at the bounds finite (see [`TanhNormal`](torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal)).
Defaults to `False`.

See also

[`torchrl.modules.TanhNormal`](torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal)