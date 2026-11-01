# SafeProbabilisticTensorDictSequential

*class*torchrl.modules.tensordict_module.SafeProbabilisticTensorDictSequential(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/probabilistic.html#SafeProbabilisticTensorDictSequential)

[`tensordict.nn.ProbabilisticTensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) subclass that accepts a `TensorSpec` as argument to control the output domain.

Similarly to `TensorDictSequential`, but enforces that the final module in the
sequence is an `ProbabilisticTensorDictModule` and also exposes `get_dist`
method to recover the distribution object from the `ProbabilisticTensorDictModule`

Parameters:

- **modules** (*iterable**of**TensorDictModules*) - ordered sequence of TensorDictModule
instances, terminating in ProbabilisticTensorDictModule, to be run
sequentially.
- **partial_tolerant** (*bool**,**optional*) - if `True`, the input tensordict can miss some
of the input keys. If so, the only modules that will be executed are those
which can be executed given the keys that are present. Also, if the input
tensordict is a lazy stack of tensordicts AND if partial_tolerant is
`True` AND if the stack does not have the required keys, then
TensorDictSequential will scan through the sub-tensordicts looking for those
that have the required keys, if any.