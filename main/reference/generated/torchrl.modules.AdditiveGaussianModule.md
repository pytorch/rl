# AdditiveGaussianModule

*class*torchrl.modules.AdditiveGaussianModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#AdditiveGaussianModule)

Additive Gaussian PO module.

Parameters:

- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - the spec used for sampling actions. The sampled
action will be projected onto the valid action space once explored.
Can be `None` for delayed initialization, in which case the spec
must be set via the `spec` property setter before calling
`forward()`.
default: None
- **sigma_init** (*scalar**,**optional*) - initial epsilon value.
default: 1.0
- **sigma_end** (*scalar**,**optional*) - final epsilon value.
default: 0.1
- **annealing_num_steps** (*int**,**optional*) - number of steps it will take for
sigma to reach the `sigma_end` value.
default: 1000
- **mean** (`float`, optional) - mean of each output element's normal distribution.
default: 0.0
- **std** (`float`, optional) - standard deviation of each output element's normal distribution.
default: 1.0

Keyword Arguments:

- **action_key** (*NestedKey**,**optional*) - if the policy module has more than one output key,
its output spec will be of type Composite. One needs to know where to
find the action spec.
default: "action"
- **safe** (*bool*) - if `True`, actions that are out of bounds given the action specs will be projected in the space
given the `TensorSpec.project` heuristic.
default: False
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device where the buffers have to be stored.

Note

It is
crucial to incorporate a call to `step()` in the training loop
to update the exploration factor.
Since it is not easy to capture this omission no warning or exception
will be raised if this is omitted!

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#AdditiveGaussianModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

step(*frames: int = 1*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#AdditiveGaussianModule.step)

A step of sigma decay.

After self.annealing_num_steps calls to this method, calls result in no-op.

Parameters:

**frames** (*int*) - number of frames since last step. Defaults to `1`.