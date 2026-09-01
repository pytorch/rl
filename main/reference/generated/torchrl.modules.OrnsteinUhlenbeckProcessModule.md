# OrnsteinUhlenbeckProcessModule

*class*torchrl.modules.OrnsteinUhlenbeckProcessModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#OrnsteinUhlenbeckProcessModule)

Ornstein-Uhlenbeck exploration policy module.

Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", [https://arxiv.org/pdf/1509.02971.pdf](https://arxiv.org/pdf/1509.02971.pdf).

The OU exploration is to be used with continuous control policies and introduces a auto-correlated exploration
noise. This enables a sort of 'structured' exploration.

Noise equation:

\[noise_t = noise_{t-1} + \theta * (mu - noise_{t-1}) * dt + \sigma_t * \sqrt{dt} * W\]

Sigma equation:

\[\sigma_t = max(\sigma^{min, (-(\sigma_{t-1} - \sigma^{min}) / (n^{\text{steps annealing}}) * n^{\text{steps}} + \sigma))\]

To keep track of the steps and noise from sample to sample, an `"ou_prev_noise{id}"` and `"ou_steps{id}"` keys
will be written in the input/output tensordict. It is expected that the tensordict will be zeroed at reset,
indicating that a new trajectory is being collected. If not, and is the same tensordict is used for consecutive
trajectories, the step count will keep on increasing across rollouts. Note that the collector classes take care of
zeroing the tensordict at reset time.

Note

It is
crucial to incorporate a call to `step()` in the training loop
to update the exploration factor.
Since it is not easy to capture this omission no warning or exception
will be raised if this is omitted!

Parameters:

- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - the spec used for sampling actions. The sampled
action will be projected onto the valid action space once explored.
- **eps_init** (*scalar*) - initial epsilon value, determining the amount of noise to be added.
default: 1.0
- **eps_end** (*scalar*) - final epsilon value, determining the amount of noise to be added.
default: 0.1
- **annealing_num_steps** (*int*) - number of steps it will take for epsilon to reach the eps_end value.
default: 1000
- **theta** (*scalar*) - theta factor in the noise equation
default: 0.15
- **mu** (*scalar*) - OU average (mu in the noise equation).
default: 0.0
- **sigma** (*scalar*) - sigma value in the sigma equation.
default: 0.2
- **dt** (*scalar*) - dt in the noise equation.
default: 0.01
- **x0** (*Tensor**,**ndarray**,**optional*) - initial value of the process.
default: 0.0
- **sigma_min** (*number**,**optional*) - sigma_min in the sigma equation.
default: None
- **n_steps_annealing** (*int*) - number of steps for the sigma annealing.
default: 1000

Keyword Arguments:

- **action_key** (*NestedKey**,**optional*) - key of the action to be modified.
default: "action"
- **is_init_key** (*NestedKey**,**optional*) - key where to find the is_init flag used to reset the noise steps.
default: "is_init"
- **safe** (*boolean**,**optional*) - if False, the TensorSpec can be None. If it
is set to False but the spec is passed, the projection will still
happen.
Default is True.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device where the buffers have to be stored.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictSequential
>>> from torchrl.data import Bounded
>>> from torchrl.modules import OrnsteinUhlenbeckProcessModule, Actor
>>> torch.manual_seed(0)
>>> spec = Bounded(-1, 1, torch.Size([4]))
>>> module = torch.nn.Linear(4, 4, bias=False)
>>> policy = Actor(module=module, spec=spec)
>>> ou = OrnsteinUhlenbeckProcessModule(spec=spec)
>>> explorative_policy = TensorDictSequential(policy, ou)
>>> td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
>>> print(explorative_policy(td))
TensorDict(
 fields={
 _ou_prev_noise: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 _ou_steps: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
 action: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#OrnsteinUhlenbeckProcessModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

step(*frames: int = 1*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#OrnsteinUhlenbeckProcessModule.step)

Updates the eps noise factor.

Parameters:

**frames** (*int*) - number of frames of the current batch (corresponding to the number of updates to be made).