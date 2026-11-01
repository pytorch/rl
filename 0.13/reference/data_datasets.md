# Datasets

TorchRL provides dataset utilities for offline RL and data management.

## TorchRL Episode Data (TED) Format

The TED format is TorchRL's standard data layout for offline RL datasets. It structures
trajectories as nested tensors where each element contains the full trajectory data,
making it efficient for sequence-based sampling and training.

| [`datasets.AtariDQNExperienceReplay`](generated/torchrl.data.datasets.AtariDQNExperienceReplay.html#torchrl.data.datasets.AtariDQNExperienceReplay)(dataset_id) | Atari DQN Experience replay class. |
| --- | --- |
| [`datasets.D4RLExperienceReplay`](generated/torchrl.data.datasets.D4RLExperienceReplay.html#torchrl.data.datasets.D4RLExperienceReplay)(dataset_id, ...) | An Experience replay class for D4RL. |
| [`datasets.GenDGRLExperienceReplay`](generated/torchrl.data.datasets.GenDGRLExperienceReplay.html#torchrl.data.datasets.GenDGRLExperienceReplay)(dataset_id) | Gen-DGRL Experience Replay dataset. |
| [`datasets.MinariExperienceReplay`](generated/torchrl.data.datasets.MinariExperienceReplay.html#torchrl.data.datasets.MinariExperienceReplay)(dataset_id, ...) | Minari Experience replay dataset. |
| [`datasets.OpenMLExperienceReplay`](generated/torchrl.data.datasets.OpenMLExperienceReplay.html#torchrl.data.datasets.OpenMLExperienceReplay)(name, batch_size) | An experience replay for OpenML data. |
| [`datasets.OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay)(dataset_id[, ...]) | Open X-Embodiment datasets experience replay. |
| [`datasets.RobosetExperienceReplay`](generated/torchrl.data.datasets.RobosetExperienceReplay.html#torchrl.data.datasets.RobosetExperienceReplay)(dataset_id, ...) | Roboset experience replay dataset. |
| [`datasets.VD4RLExperienceReplay`](generated/torchrl.data.datasets.VD4RLExperienceReplay.html#torchrl.data.datasets.VD4RLExperienceReplay)(dataset_id, ...) | V-D4RL experience replay dataset. |