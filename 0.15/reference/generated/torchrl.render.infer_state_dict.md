# infer_state_dict

torchrl.render.infer_state_dict(*payload: Any*, *key: str | None = None*) → Mapping[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/render/checkpoint.html#infer_state_dict)

Infers a model state dict from common checkpoint payload layouts.

Parameters:

- **payload** - Checkpoint payload.
- **key** - Explicit state-dict key to read from mapping payloads.

Returns:

A mapping from parameter names to tensors.