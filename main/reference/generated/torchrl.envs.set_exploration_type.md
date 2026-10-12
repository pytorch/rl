# set_exploration_type

torchrl.envs.set_exploration_type(*type: InteractionType | str | None = InteractionType.DETERMINISTIC*) → None

Sets the interaction type of all probabilistic modules inside a block of code, or inside a decorated function.

Inside the block, `interaction_type()`
returns `type`, and every `ProbabilisticTensorDictModule` uses it
instead of its `default_interaction_type`. When the block ends, the
previous value comes back.

Parameters:

**type** (*InteractionType**or**str**,**optional*) - the interaction type, for
example `InteractionType.RANDOM` or `"random"` (in any case).
`None` makes each module use its `default_interaction_type`.
Defaults to `InteractionType.DETERMINISTIC`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... InteractionType,
... ProbabilisticTensorDictModule,
... set_interaction_type,
... )
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
... )
>>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
>>> module(td)["action"] # default_interaction_type is DETERMINISTIC: the mean
tensor([0., 0., 0.])
>>> _ = torch.manual_seed(0)
>>> with set_interaction_type("random"):
... module(td)["action"]
tensor([ 1.5410, -0.2934, -2.1788])
```

It also works as a decorator:

```
>>> @set_interaction_type(InteractionType.MODE)
... def act(td):
... return module(td)
>>> act(td)["action"]
tensor([0., 0., 0.])
```