# export_onnx_policy

torchrl.modules.export_onnx_policy(*policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *path: str | Path*, *example: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *out_keys: Sequence[[NestedKey](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NestedKey.html#tensordict.NestedKey)] | None = None*, *opset_version: int = 18*) → Path[[source]](../../_modules/torchrl/modules/utils/export.html#export_onnx_policy)

Export a policy to ONNX by tracing it with [`torch.onnx.export()`](https://docs.pytorch.org/docs/stable/onnx_export.html#torch.onnx.export).

The graph runs the policy on one unbatched step. Its inputs are
`policy.in_keys` and its outputs are `out_keys`, in that order, each
named after its key with nested keys joined by `"_"`. A recurrent policy
takes its state and `is_init` as inputs and returns the next state. Env
transforms are not exported, so observation normalization must be part of
`policy`. Requires the `onnx` extra (`pip install torchrl[onnx]`).

The policy is traced as configured and nothing is overridden. Call
`policy.eval()` first: a policy in training mode is exported with its
training behavior, such as dropout, and PyTorch warns about it. Set the
exploration type and recurrent mode around the call, as in the example.

Parameters:

- **policy** (*TensorDictModuleBase*) - the policy to export.
- **path** (*str**or**Path*) - file to write.
- **example** (*TensorDictBase*) - unbatched input to trace with, holding every
entry of `policy.in_keys`.

Keyword Arguments:

- **out_keys** (*sequence**of**NestedKey**,**optional*) - entries the graph returns,
in order. Defaults to `"action"` followed by `("next", key)` for
every input key the policy also writes under `"next"`, i.e. its
recurrent state.
- **opset_version** (*int**,**optional*) - ONNX opset version. Defaults to `18`.

Returns:

The path written.

Raises:

- **ValueError** - if `example` has a batch dimension, or if two input or
 output keys get the same name.
- **RuntimeError** - if the exported graph does not hold one distinct input
 or output per key, so it cannot be named after them.

Examples

```
>>> import torch
>>> from tensordict.nn import (
... AddStateIndependentNormalScale,
... TensorDictModule,
... TensorDictSequential,
... )
>>> from torch import nn
>>> from torchrl.envs import ObservationNorm
>>> from torchrl.envs.utils import ExplorationType, set_exploration_type
>>> from torchrl.modules import (
... export_onnx_policy,
... LSTMModule,
... MLP,
... ProbabilisticActor,
... set_recurrent_mode,
... TanhNormal,
... )
>>> lstm = LSTMModule(input_size=61, hidden_size=64, in_key="observation_norm", out_key="features")
>>> head = TensorDictModule(
... nn.Sequential(MLP(in_features=64, out_features=14, num_cells=[64]), AddStateIndependentNormalScale(14)),
... in_keys=["features"],
... out_keys=["loc", "scale"],
... )
>>> actor = ProbabilisticActor(
... TensorDictSequential(lstm, head),
... in_keys=["loc", "scale"],
... distribution_class=TanhNormal,
... )
>>> norm = ObservationNorm(
... loc=torch.zeros(61), scale=torch.ones(61), in_keys=["observation"], out_keys=["observation_norm"]
... )
>>> policy = TensorDictSequential(norm, actor).eval()
>>> example = lstm.make_tensordict_primer().primers.zero()
>>> example["observation"] = torch.zeros(61)
>>> example["is_init"] = torch.zeros(1, dtype=torch.bool)
>>> with set_exploration_type(ExplorationType.DETERMINISTIC), set_recurrent_mode(False): 
... export_onnx_policy(policy, "walk.onnx", example)
PosixPath('walk.onnx')
```