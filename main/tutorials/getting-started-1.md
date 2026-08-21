Note

Go to the end
to download the full example code.

# Get started with TorchRL's modules

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

Reinforcement Learning is designed to create policies that can effectively
tackle specific tasks. Policies can take various forms, from a differentiable
map transitioning from the observation space to the action space, to a more
ad-hoc method like an argmax over a list of values computed for each possible
action. Policies can be deterministic or stochastic, and may incorporate
complex elements such as Recurrent Neural Networks (RNNs) or transformers.

Accommodating all these scenarios can be quite intricate. In this succinct
tutorial, we will delve into the core functionality of TorchRL in terms of
policy construction. We will primarily focus on stochastic and Q-Value
policies in two common scenarios: using a Multi-Layer Perceptron (MLP) or
a Convolutional Neural Network (CNN) as backbones.

## TensorDictModules

Similar to how environments interact with instances of
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), the modules used to represent policies and
value functions also do the same. The core idea is simple: encapsulate a
standard [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) (or any other function) within a class
that knows which entries need to be read and passed to the module, and then
records the results with the assigned entries. To illustrate this, we will
use the simplest policy possible: a deterministic map from the observation
space to the action space. For maximum generality, we will use a
[`LazyLinear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) module with the Pendulum environment we
instantiated in the previous tutorial.

```
import torch

from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
 module,
 in_keys=["observation"],
 out_keys=["action"],
)
```

This is all that's required to execute our policy! The use of a lazy module
allows us to bypass the need to fetch the shape of the observation space, as
the module will automatically determine it. This policy is now ready to be
run in the environment:

```
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

## Specialized wrappers

To simplify the incorporation of `Actor`,
# [`ProbabilisticActor`](../reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor),
# [`ActorValueOperator`](../reference/generated/torchrl.modules.ActorValueOperator.html#torchrl.modules.ActorValueOperator) or
# [`ActorCriticOperator`](../reference/generated/torchrl.modules.ActorCriticOperator.html#torchrl.modules.ActorCriticOperator).
For example, [`Actor`](../reference/generated/torchrl.modules.tensordict_module.Actor.html#torchrl.modules.tensordict_module.Actor) provides
default values for the `in_keys` and `out_keys`, making integration
with many common environments straightforward:

```
from torchrl.modules import Actor

policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

The list of available specialized TensorDictModules is available in the
[API reference](../reference/modules.html#tdmodules).

## Networks

TorchRL also provides regular modules that can be used without recurring to
tensordict features. The two most common networks you will encounter are
the [`MLP`](../reference/generated/torchrl.modules.MLP.html#torchrl.modules.MLP) and the [`ConvNet`](../reference/generated/torchrl.modules.ConvNet.html#torchrl.modules.ConvNet)
(CNN) modules. We can substitute our policy module with one of these:

```
from torchrl.modules import MLP

module = MLP(
 out_features=env.action_spec.shape[-1],
 num_cells=[32, 64],
 activation_class=torch.nn.Tanh,
)
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
```

TorchRL also supports RNN-based policies. Since this is a more technical
topic, it is treated in [a separate tutorial](dqn_with_rnn.html#rnn-tuto).

## Probabilistic policies

Policy-optimization algorithms like
[PPO](https://arxiv.org/abs/1707.06347) require the policy to be
stochastic: unlike in the examples above, the module now encodes a map from
the observation space to a parameter space encoding a distribution over the
possible actions. TorchRL facilitates the design of such modules by grouping
under a single class the various operations such as building the distribution
from the parameters, sampling from that distribution and retrieving the
log-probability. Here, we'll be building an actor that relies on a regular
normal distribution using three components:

- An [`MLP`](../reference/generated/torchrl.modules.MLP.html#torchrl.modules.MLP) backbone reading observations of size
`[3]` and outputting a single tensor of size `[2]`;
- A [`NormalParamExtractor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor) module that
will split this output on two chunks, a mean and a standard deviation of
size `[1]`;
- A [`ProbabilisticActor`](../reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor) that will
read those parameters as `in_keys`, create a distribution with them and
populate our tensordict with samples and log-probabilities.

```
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor

backbone = MLP(in_features=3, out_features=2)
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
 td_module,
 in_keys=["loc", "scale"],
 out_keys=["action"],
 distribution_class=Normal,
 return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 action_log_prob: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 loc: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

There are a few things to note about this rollout:

- Since we asked for it during the construction of the actor, the
log-probability of the actions given the distribution at that time is
also written. This is necessary for algorithms like PPO.
- The parameters of the distribution are returned within the output
tensordict too under the `"loc"` and `"scale"` entries.

You can control the sampling of the action to use the expected value or
other properties of the distribution instead of using random samples if
your application requires it. This can be controlled via the
[`set_exploration_type()`](../reference/generated/torchrl.envs.set_exploration_type.html#torchrl.envs.set_exploration_type) function:

```
from torchrl.envs.utils import ExplorationType, set_exploration_type

with set_exploration_type(ExplorationType.DETERMINISTIC):
 # takes the mean as action
 rollout = env.rollout(max_steps=10, policy=policy)
with set_exploration_type(ExplorationType.RANDOM):
 # Samples actions according to the dist
 rollout = env.rollout(max_steps=10, policy=policy)
```

Check the `default_interaction_type` keyword argument in
the docstrings to know more.

## Exploration

Stochastic policies like this somewhat naturally trade off exploration and
exploitation, but deterministic policies won't. Fortunately, TorchRL can
also palliate to this with its exploration modules.
We will take the example of the [`EGreedyModule`](../reference/generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule)
exploration module (check also
[`AdditiveGaussianModule`](../reference/generated/torchrl.modules.AdditiveGaussianModule.html#torchrl.modules.AdditiveGaussianModule) and
[`OrnsteinUhlenbeckProcessModule`](../reference/generated/torchrl.modules.OrnsteinUhlenbeckProcessModule.html#torchrl.modules.OrnsteinUhlenbeckProcessModule)).
To see this module in action, let's revert to a deterministic policy:

```
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

policy = Actor(MLP(3, 1, num_cells=[32, 64]))
```

Our \(\epsilon\)-greedy exploration module will usually be customized
with a number of annealing frames and an initial value for the
\(\epsilon\) parameter. A value of \(\epsilon = 1\) means that every
action taken is random, while \(\epsilon=0\) means that there is no
exploration at all. To anneal (i.e., decrease) the exploration factor, a call
to [`step()`](../reference/generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule.step) is required (see the last
[tutorial](getting-started-5.html#gs-first-training) for an example).

```
exploration_module = EGreedyModule(
 spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5
)
```

To build our explorative policy, we only had to concatenate the
deterministic policy module with the exploration module within a
[`TensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) module (which is the analogous
to [`Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential) in the tensordict realm).

```
exploration_policy = TensorDictSequential(policy, exploration_module)

with set_exploration_type(ExplorationType.DETERMINISTIC):
 # Turns off exploration
 rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
 # Turns on exploration
 rollout = env.rollout(max_steps=10, policy=exploration_policy)
```

Because it must be able to sample random actions in the action space, the
[`EGreedyModule`](../reference/generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule) must be equipped with the
`action_space` from the environment to know what strategy to use to
sample actions randomly.

## Q-Value actors

In some settings, the policy isn't a standalone module but is constructed on
top of another module. This is the case with **Q-Value actors**. In short, these
actors require an estimate of the action value (most of the time discrete)
and will greedily pick up the action with the highest value. In some
settings (finite discrete action space and finite discrete state space),
one can just store a 2D table of state-action pairs and pick up the
action with the highest value. The innovation brought by
[DQN](https://arxiv.org/abs/1312.5602) was to scale this up to continuous
state spaces by utilizing a neural network to encode for the `Q(s, a)`
value map. Let's consider another environment with a discrete action space
for a clearer understanding:

```
env = GymEnv("CartPole-v1")
print(env.action_spec)
```

```
OneHot(
 shape=torch.Size([2]),
 space=CategoricalBox(n=2),
 device=cpu,
 dtype=torch.int64,
 domain=discrete)
```

We build a value network that produces one value per action when it reads a
state from the environment:

```
num_actions = 2
value_net = TensorDictModule(
 MLP(out_features=num_actions, num_cells=[32, 32]),
 in_keys=["observation"],
 out_keys=["action_value"],
)
```

We can easily build our Q-Value actor by adding a
[`QValueModule`](../reference/generated/torchrl.modules.QValueModule.html#torchrl.modules.QValueModule) after our value
network:

```
from torchrl.modules import QValueModule

policy = TensorDictSequential(
 value_net, # writes action values in our tensordict
 QValueModule(spec=env.action_spec), # Reads the "action_value" entry by default
)
```

Let's check it out! We run the policy for a couple of steps and look at the
output. We should find an `"action_value"` as well as a
`"chosen_action_value"` entries in the rollout that we obtain:

```
rollout = env.rollout(max_steps=3, policy=policy)
print(rollout)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 action_value: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 chosen_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

Because it relies on the `argmax` operator, this policy is deterministic.
During data collection, we will need to explore the environment. For that,
we are using the [`EGreedyModule`](../reference/generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule)
once again:

```
policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec))

with set_exploration_type(ExplorationType.RANDOM):
 rollout_explore = env.rollout(max_steps=3, policy=policy_explore)
```

This is it for our short tutorial on building a policy with TorchRL!

There are many more things you can do with the library. A good place to start
is to look at the [API reference for modules](../reference/modules.html#ref-modules).

Next steps:

- Check how to use compound distributions with
[`CompositeDistribution`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) when the
action is composite (e.g., a discrete and a continuous action are
required by the env);
- Have a look at how you can use an RNN within the policy (a
[tutorial](dqn_with_rnn.html#rnn-tuto));
- Compare this to the usage of transformers with the Decision Transformers
examples (see the `example` directory on GitHub).

**Total running time of the script:** (0 minutes 0.072 seconds)

[`Download Jupyter notebook: getting-started-1.ipynb`](../_downloads/c4e061fa0d0af4f742988ea6303991f1/getting-started-1.ipynb)

[`Download Python source code: getting-started-1.py`](../_downloads/84ffea919d8686a89104bfcc27788f3e/getting-started-1.py)

[`Download zipped: getting-started-1.zip`](../_downloads/6a76fecc1b49e983846e88271a9db64b/getting-started-1.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)