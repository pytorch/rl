# lerobot_columns_to_tensordict

*class*torchrl.data.datasets.lerobot_columns_to_tensordict(*columns: dict[str, Any]*, ***, *key_map: dict[str, NestedKey] | None = None*)[[source]](../../_modules/torchrl/data/datasets/lerobot.html#lerobot_columns_to_tensordict)

Convert a LeRobot-style columnar dict into a canonical VLA TensorDict.

LeRobot stores per-frame data under dotted column names
(`observation.state`, `observation.images.<camera>`, `action`,
`episode_index`, `task`, ...). This builds a flat `[N]` TensorDict
using the canonical VLA key layout: proprioceptive state and images under
`observation`, the per-frame language instruction and the action at the
root, and `episode` for trajectory boundaries (see
[`validate_vla_tensordict()`](torchrl.data.vla.validate_vla_tensordict.html#torchrl.data.vla.validate_vla_tensordict)).

Parameters:

**columns** (*dict*) - mapping from LeRobot column name to a tensor (numeric
columns), a list of strings (e.g. the `task` instruction), or a
[`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef) (lazy video frames, decoded on
sampling by [`DecodeVideoTransform`](torchrl.envs.transforms.DecodeVideoTransform.html#torchrl.envs.transforms.DecodeVideoTransform)).

Keyword Arguments:

**key_map** (*dict**,**optional*) - overrides/extends `_DEFAULT_KEY_MAP`,
mapping a source column name to a target `NestedKey`.

Returns:

a flat `[N]` [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

Examples

```
>>> import torch
>>> from torchrl.data.datasets.lerobot import lerobot_columns_to_tensordict
>>> columns = {
... "observation.state": torch.zeros(4, 7),
... "observation.images.top": torch.zeros(4, 3, 8, 8, dtype=torch.uint8),
... "action": torch.zeros(4, 7),
... "episode_index": torch.tensor([0, 0, 1, 1]),
... "task": ["pick", "pick", "place", "place"],
... }
>>> td = lerobot_columns_to_tensordict(columns)
>>> td["observation", "state"].shape
torch.Size([4, 7])
>>> td["observation", "image", "top"].shape
torch.Size([4, 3, 8, 8])
>>> td.get("language_instruction").tolist()
['pick', 'pick', 'place', 'place']
```