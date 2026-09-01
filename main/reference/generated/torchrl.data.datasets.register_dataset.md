# register_dataset

torchrl.data.datasets.register_dataset(*prefix: str*, *dataset: str | Callable[[...], BaseDatasetExperienceReplay]*, ***, *replace: bool = False*) → None[[source]](../../_modules/torchrl/data/datasets/utils.html#register_dataset)

Register a dataset factory for [`load_dataset()`](torchrl.data.datasets.load_dataset.html#torchrl.data.datasets.load_dataset).

The registered prefix can then be used in strings of the form
`"<prefix>:<dataset-id>"`. The dataset factory is called as
`dataset(dataset_id, **kwargs)`.

Parameters:

- **prefix** (*str*) - source prefix used before the `":"` separator.
- **dataset** (*Callable**or**str*) - dataset factory, or an import string of the
form `"module:attribute"` resolved lazily when the prefix is used.
- **replace** (*bool**,**optional*) - if `True`, replace an existing registration.
Defaults to `False`.

Examples

```
>>> from torchrl.data.datasets import register_dataset, load_dataset
>>> class ToyDataset:
... def __init__(self, dataset_id, **kwargs):
... self.dataset_id = dataset_id
>>> register_dataset("toy", ToyDataset, replace=True)
>>> load_dataset("toy:example").dataset_id
'example'
```