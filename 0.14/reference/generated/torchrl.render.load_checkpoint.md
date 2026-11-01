# load_checkpoint

torchrl.render.load_checkpoint(*path: str | Path*, *map_location: str | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) = 'cpu'*, ***, *weights_only: bool | None = None*) → Any[[source]](../../_modules/torchrl/render/checkpoint.html#load_checkpoint)

Loads a local PyTorch checkpoint.

Parameters:

- **path** - Local checkpoint path.
- **map_location** - Device mapping passed to [`torch.load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load).
- **weights_only** - Whether payloads are restricted to safe weight-only
types. Unified checkpoints default to `True`. Legacy payloads
retain their historical `False` default for compatibility.

Returns:

The checkpoint payload.