# validate_vla_tensordict

*class*torchrl.data.vla.validate_vla_tensordict(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *instruction_key: NestedKey = 'language_instruction'*, *action_key: NestedKey = 'action'*, *image_key: NestedKey = ('observation', 'image')*, *state_key: NestedKey = ('observation', 'state')*, *require_instruction: bool = True*, *require_action: bool = True*, *require_perception: bool = True*, *check_finite: bool = True*, *raise_on_error: bool = True*)[[source]](../../_modules/torchrl/data/vla/schema.html#validate_vla_tensordict)

Validate that a tensordict follows the canonical VLA schema.

The check is intentionally permissive: it verifies the presence of the
keys a VLA pipeline relies on and that action tensors are finite, without
constraining shapes beyond what is necessary.

Parameters:

**tensordict** (*TensorDictBase*) - the tensordict to validate.

Keyword Arguments:

- **instruction_key** (*NestedKey*) - language-instruction key.
Defaults to `("observation", "language_instruction")`.
- **action_key** (*NestedKey*) - action key. Defaults to `"action"`.
- **image_key** (*NestedKey*) - image key.
Defaults to `("observation", "image")`.
- **state_key** (*NestedKey*) - proprioceptive-state key.
Defaults to `("observation", "state")`.
- **require_instruction** (*bool*) - if `True`, a missing instruction is an
error. Defaults to `True`.
- **require_action** (*bool*) - if `True`, a missing action is an error.
Defaults to `True`.
- **require_perception** (*bool*) - if `True`, at least one of the image or
state keys must be present. Defaults to `True`.
- **check_finite** (*bool*) - if `True`, float action tensors must be finite.
Defaults to `True`.
- **raise_on_error** (*bool*) - if `True` (default), raise a `ValueError`
when any issue is found; otherwise return the list of issues.

Returns:

a list of human-readable issue strings (empty if the tensordict is
valid). When `raise_on_error` is `True` a non-empty list raises a
`ValueError` instead of being returned.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData, TensorDict
>>> from torchrl.data.vla import validate_vla_tensordict
>>> td = TensorDict(
... {
... "observation": {
... "image": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
... },
... "language_instruction": NonTensorData("pick the cube"),
... "action": torch.zeros(2, 7),
... },
... batch_size=[2],
... )
>>> validate_vla_tensordict(td)
[]
```