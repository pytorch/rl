# DreamerDecoder

torchrl.envs.model_based.dreamer.DreamerDecoder(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/model_based/dreamer.html#DreamerDecoder)

A transform to record the decoded observations in Dreamer.

Examples

```
>>> model_based_env = DreamerEnv(...)
>>> model_based_env_eval = model_based_env.append_transform(DreamerDecoder())
```