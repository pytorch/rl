# MCAdvantageSelector

*class*torchrl.objectives.llm.MCAdvantageSelector(*strategy: Literal['first', 'uniform', 'balanced'] = 'balanced'*, ***, *max_combinations: int = 100000*, *in_keys: list[NestedKey] | None = None*)[[source]](../../_modules/torchrl/objectives/llm/grpo.html#MCAdvantageSelector)

Select trajectories from an oversampled Monte-Carlo advantage group.

`MCAdvantage` can collect more candidate trajectories for a group than
the number used for the GRPO update. This selector chooses the subset that
should be written to storage and trained on. The default `"balanced"`
strategy keeps the historical behavior when there is no oversampling, and
when oversampling is enabled it tries to pick a subset whose mean return
lies inside the dynamic-sampling bounds.

Parameters:

- **strategy** (*str**,**optional*) - Selection strategy. `"first"` selects the
first `group_size` candidates, matching the non-oversampled
behavior. `"uniform"` sorts candidates by return and samples
roughly uniformly across that order. `"balanced"` searches for
a subset that passes `keep_return_bounds` and is closest to the
middle of the accepted interval. Defaults to `"balanced"`.
- **max_combinations** (*int**,**optional*) - Maximum exact combinations to score
for `"balanced"` selection. Larger candidate pools fall back to
a deterministic greedy strategy. Defaults to `100_000`.
- **in_keys** (*list**of**NestedKey**,**optional*) - Candidate keys consumed by the
selector. Defaults to `["return"]`. `MCAdvantage` passes a
candidate-level tensordict with one entry per candidate trajectory,
containing `"return"` and a lazy-stacked `"trajectories"`
tensordict with the full candidate trajectories. Subclasses can
set this argument and override `select()` to implement custom
metadata- or trajectory-based selection.

Examples

```
>>> import torch
>>> from torchrl.objectives.llm import MCAdvantageSelector
>>> from tensordict import TensorDict
>>> selector = MCAdvantageSelector()
>>> selector.select(
... TensorDict({"return": torch.tensor([0.0, 0.0, 0.0, 1.0])}, [4]),
... group_size=2,
... keep_return_bounds=(0.1, 0.9),
... )
[0, 3]
```

select(*candidates: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *group_size: int*, *keep_return_bounds: tuple[float, float] | None = None*) → list[int] | None[[source]](../../_modules/torchrl/objectives/llm/grpo.html#MCAdvantageSelector.select)

Select candidate indices.

Parameters:

- **candidates** (*TensorDictBase*) - Candidate-level tensordict with one
entry per candidate trajectory. The default selector reads the
first `in_keys` entry as a scalar value per candidate.
- **group_size** (*int*) - Number of trajectories to select.
- **keep_return_bounds** (*tuple**of**float**,**optional*) - Accepted exclusive
mean-return interval. If supplied and no valid subset is found,
`None` is returned.

Returns:

Selected candidate indices, or `None` when the

candidate group should be skipped.

Return type:

list[int] or None