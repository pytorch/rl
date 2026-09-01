# SamplerWithoutReplacement

*class*torchrl.data.replay_buffers.SamplerWithoutReplacement(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#SamplerWithoutReplacement)

A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

Parameters:

- **drop_last** (*bool**,**optional*) - if `True`, the last incomplete sample (if any) will be dropped.
If `False`, this last sample will be kept and (unlike with torch dataloaders)
completed with other samples from a fresh indices permutation.
Defaults to `False`.
- **shuffle** (*bool**,**optional*) - if `False`, the items are not randomly
permuted. This enables to iterate over the replay buffer in the
order the data was collected. Defaults to `True`.

*Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
(as we can't generally keep track of which samples have been sampled before and which haven't).

Similarly, it is expected that the storage content remains the same in between two calls,
but this is not enforced.

When the sampler reaches the end of the list of available indices, a new sample order
will be generated and the resulting indices will be completed with this new draw, which
can lead to duplicated indices, unless the `drop_last` argument is set to `True`.