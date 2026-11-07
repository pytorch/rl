# load_dataset

torchrl.data.datasets.load_dataset(*dataset_id: str*, ***kwargs*) → BaseDatasetExperienceReplay[[source]](../../_modules/torchrl/data/datasets/utils.html#load_dataset)

Parse a dataset ID string and return the registered dataset object.

Built-in prefixes include `"atari"`, `"atari_dqn"`, `"d4rl"`,
`"gen_dgrl"`, `"lerobot"`, `"minari"`, `"openml"`, `"openx"`,
`"roboset"`, and `"vd4rl"`. Additional prefixes can be installed with
[`register_dataset()`](torchrl.data.datasets.register_dataset.html#torchrl.data.datasets.register_dataset).

Parameters:

- **dataset_id** (*str*) - a prefixed dataset identifier, e.g.
`"minari:mujoco/hopper/expert-v0"` or
`"d4rl:halfcheetah-medium-v2"`.
- ****kwargs** - forwarded to the dataset constructor.

Returns:

the constructed dataset object.

Return type:

BaseDatasetExperienceReplay

Examples

```
>>> from torchrl.data.datasets import register_dataset
>>> class ToyDataset:
... def __init__(self, dataset_id, **kwargs):
... self.dataset_id = dataset_id
>>> register_dataset("toy", ToyDataset, replace=True)
>>> load_dataset("toy:example").dataset_id
'example'
```