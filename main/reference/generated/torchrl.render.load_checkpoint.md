# load_checkpoint

torchrl.render.load_checkpoint(*path: str | Path*, *map_location: str | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) = 'cpu'*) → Any[[source]](../../_modules/torchrl/render/checkpoint.html#load_checkpoint)

Loads a local PyTorch checkpoint.

Parameters:

- **path** - Local checkpoint path.
- **map_location** - Device mapping passed to [`torch.load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load).

Returns:

The checkpoint payload.