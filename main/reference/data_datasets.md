# Datasets

TorchRL provides dataset utilities for offline RL and data management.

## TorchRL Episode Data (TED) Format

The TED format is TorchRL's standard data layout for offline RL datasets. It structures
trajectories as nested tensors where each element contains the full trajectory data,
making it efficient for sequence-based sampling and training.

## Dataset loading registry

Many offline datasets can be constructed from a compact string identifier with
[`load_dataset()`](generated/torchrl.data.datasets.load_dataset.html#torchrl.data.datasets.load_dataset). The identifier is split as
`"<source>:<dataset-id>"`. The `source` prefix selects a registered dataset
factory and the remainder is forwarded as the first constructor argument, with
any extra keyword arguments forwarded unchanged.

TorchRL registers the built-in dataset families at import time, including
`"atari"`, `"atari_dqn"`, `"d4rl"`, `"gen_dgrl"`, `"lerobot"`,
`"minari"`, `"openml"`, `"openx"`, `"roboset"`, and `"vd4rl"`.
For example:

```
from torchrl.data.datasets import load_dataset

dataset = load_dataset("d4rl:halfcheetah-medium-v2", batch_size=256)
minari_dataset = load_dataset(
 "minari:mujoco/hopper/expert-v0",
 batch_size=256,
 split_trajs=True,
)
```

Projects can add their own sources with
[`register_dataset()`](generated/torchrl.data.datasets.register_dataset.html#torchrl.data.datasets.register_dataset). A registered factory can be a
callable or a lazy import string of the form `"module:attribute"`. Factories
are called as `factory(dataset_id, **kwargs)`.

```
from torchrl.data.datasets import load_dataset, register_dataset

class MyDataset:
 def __init__(self, dataset_id, *, batch_size):
 self.dataset_id = dataset_id
 self.batch_size = batch_size

register_dataset("my_backend", MyDataset)
dataset = load_dataset("my_backend:my-task-v0", batch_size=128)

register_dataset(
 "my_lazy_backend",
 "my_package.datasets:MyDataset",
)
```

| [`datasets.AtariDQNExperienceReplay`](generated/torchrl.data.datasets.AtariDQNExperienceReplay.html#torchrl.data.datasets.AtariDQNExperienceReplay)(dataset_id) | Atari DQN Experience replay class. |
| --- | --- |
| [`datasets.D4RLExperienceReplay`](generated/torchrl.data.datasets.D4RLExperienceReplay.html#torchrl.data.datasets.D4RLExperienceReplay)(dataset_id, ...) | An Experience replay class for D4RL. |
| [`datasets.GenDGRLExperienceReplay`](generated/torchrl.data.datasets.GenDGRLExperienceReplay.html#torchrl.data.datasets.GenDGRLExperienceReplay)(dataset_id) | Gen-DGRL Experience Replay dataset. |
| [`datasets.LeRobotExperienceReplay`](generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay)(repo_id, *) | Experience replay over a [LeRobot](https://github.com/huggingface/lerobot) dataset. |
| [`datasets.MinariExperienceReplay`](generated/torchrl.data.datasets.MinariExperienceReplay.html#torchrl.data.datasets.MinariExperienceReplay)(dataset_id, ...) | Minari Experience replay dataset. |
| [`datasets.OpenMLExperienceReplay`](generated/torchrl.data.datasets.OpenMLExperienceReplay.html#torchrl.data.datasets.OpenMLExperienceReplay)(name, batch_size) | An experience replay for OpenML data. |
| [`datasets.OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay)(dataset_id[, ...]) | Open X-Embodiment datasets experience replay. |
| [`datasets.RobosetExperienceReplay`](generated/torchrl.data.datasets.RobosetExperienceReplay.html#torchrl.data.datasets.RobosetExperienceReplay)(dataset_id, ...) | Roboset experience replay dataset. |
| [`datasets.VD4RLExperienceReplay`](generated/torchrl.data.datasets.VD4RLExperienceReplay.html#torchrl.data.datasets.VD4RLExperienceReplay)(dataset_id, ...) | V-D4RL experience replay dataset. |

| [`datasets.lerobot_columns_to_tensordict`](generated/torchrl.data.datasets.lerobot_columns_to_tensordict.html#torchrl.data.datasets.lerobot_columns_to_tensordict)(...) | Convert a LeRobot-style columnar dict into a canonical VLA TensorDict. |
| --- | --- |

| [`datasets.load_dataset`](generated/torchrl.data.datasets.load_dataset.html#torchrl.data.datasets.load_dataset)(dataset_id, **kwargs) | Parse a dataset ID string and return the registered dataset object. |
| --- | --- |
| [`datasets.register_dataset`](generated/torchrl.data.datasets.register_dataset.html#torchrl.data.datasets.register_dataset)(prefix, dataset, *) | Register a dataset factory for `load_dataset()`. |