# SamplerEnsemble

*class*torchrl.data.replay_buffers.SamplerEnsemble(**args*, ***kwargs*)

An ensemble of samplers.

This class is designed to work with [`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble).
It contains the samplers as well as the sampling strategy hyperparameters.

Parameters:

**samplers** (*sequence**of*[*Sampler*](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)) - the samplers to make the composite sampler.

Keyword Arguments:

- **p** (list, tensor of probabilities, or `"sampleable"`, optional) - if
provided, indicates the weights of each dataset during sampling.
`"sampleable"` recomputes weights from the number of records or
valid slice windows currently available in each member and excludes
members that cannot provide a batch.
- **sample_from_all** (*bool**,**optional*) - if `True`, each dataset will be sampled
from. This is not compatible with the `p` argument. Defaults to `False`.
- **num_buffer_sampled** (*int**,**optional*) - the number of buffers to sample.
if `sample_from_all=True`, this has no effect, as it defaults to the
number of buffers. If `sample_from_all=False`, buffers will be
sampled according to the probabilities `p`.

Warning

The indices provided in the info dictionary are placed in a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with
keys `index` and `buffer_ids` that allow the upper [`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble)
and `StorageEnsemble` objects to retrieve the data.
This format is different from with other samplers which usually return indices
as regular tensors.

can_sample(*storage: [StorageEnsemble](torchrl.data.replay_buffers.StorageEnsemble.html#torchrl.data.replay_buffers.StorageEnsemble)*, *batch_size: int*) → bool[[source]](../../_modules/torchrl/data/replay_buffers/samplers/ensemble.html#SamplerEnsemble.can_sample)

Returns whether the selected ensemble strategy can serve a batch.