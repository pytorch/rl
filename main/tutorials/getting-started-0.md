Note

Go to the end
to download the full example code.

# Get started with Environments, TED and transforms

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

Welcome to the getting started tutorials!

Below is the list of the topics we will be covering.

- Environments, TED and transforms;
- [TorchRL's modules](getting-started-1.html#gs-modules);
- [Losses and optimization](getting-started-2.html#gs-optim);
- [Data collection and storage](getting-started-3.html#gs-storage);
- [TorchRL's logging API](getting-started-4.html#gs-logging).

If you are in a hurry, you can jump straight away to the last tutorial,
[Your own first training loop](getting-started-5.html#gs-first-training), from where you can
backtrack every other "Getting Started" tutorial if things are not clear or
if you want to learn more about a specific topic!

## Environments in RL

The standard RL (Reinforcement Learning) training loop involves a model,
also known as a policy, which is trained to accomplish a task within a
specific environment. Often, this environment is a simulator that accepts
actions as input and produces an observation along with some metadata as
output.

In this document, we will explore the environment API of TorchRL: we will
learn how to create an environment, interact with it, and understand the
data format it uses.

## Creating an environment

In essence, TorchRL does not directly provide environments, but instead
offers wrappers for other libraries that encapsulate the simulators. The
`envs` module can be viewed as a provider for a generic
environment API, as well as a central hub for simulation backends like
[gym](https://arxiv.org/abs/1606.01540) ([`GymEnv`](../reference/generated/torchrl.envs.GymEnv.html#torchrl.envs.GymEnv)),
[Brax](https://arxiv.org/abs/2106.13281) ([`BraxEnv`](../reference/generated/torchrl.envs.BraxEnv.html#torchrl.envs.BraxEnv))
or [DeepMind Control Suite](https://arxiv.org/abs/1801.00690)
([`DMControlEnv`](../reference/generated/torchrl.envs.DMControlEnv.html#torchrl.envs.DMControlEnv)).

Creating your environment is typically as straightforward as the underlying
backend API allows. Here's an example using gym:

```
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
```

## Running an environment

Environments in TorchRL have two crucial methods:
[`reset()`](../reference/generated/torchrl.envs.EnvBase.html#id1), which initiates
an episode, and [`step()`](../reference/generated/torchrl.envs.EnvBase.html#id4), which executes an
action selected by the actor.
In TorchRL, environment methods read and write
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances.
Essentially, [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) is a generic key-based data
carrier for tensors.
The benefit of using TensorDict over plain tensors is that it enables us to
handle simple and complex data structures interchangeably. As our function
signatures are very generic, it eliminates the challenge of accommodating
different data formats. In simpler terms, after this brief tutorial,
you will be capable of operating on both simple and highly complex
environments, as their user-facing API is identical and simple!

Let's put the environment into action and see what a tensordict instance
looks like:

```
reset = env.reset()
print(reset)
```

```
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Now let's take a random action in the action space. First, sample the action:

```
reset_with_action = env.rand_action(reset)
print(reset_with_action)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This tensordict has the same structure as the one obtained from
[`EnvBase()`](../reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) with an additional `"action"` entry.
You can access the action easily, like you would do with a regular
dictionary:

```
print(reset_with_action["action"])
```

```
tensor([1.1864])
```

We now need to pass this action to the environment.
We'll be passing the entire tensordict to the `step` method, since there
might be more than one tensor to be read in more advanced cases like
Multi-Agent RL or stateless environments:

```
stepped_data = env.step(reset_with_action)
print(stepped_data)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Again, this new tensordict is identical to the previous one except for the
fact that it has a `"next"` entry (itself a tensordict!) containing the
observation, reward and done state resulting from
our action.

We call this format TED, for
[TorchRL Episode Data format](../reference/data_datasets.html#ted-format). It is
the ubiquitous way of representing data in the library, both dynamically like
here, or statically with offline datasets.

The last bit of information you need to run a rollout in the environment is
how to bring that `"next"` entry at the root to perform the next step.
TorchRL provides a dedicated [`step_mdp()`](../reference/generated/torchrl.envs.step_mdp.html#torchrl.envs.step_mdp) function
that does just that: it filters out the information you won't need and
delivers a data structure corresponding to your observation after a step in
the Markov Decision Process, or MDP.

```
from torchrl.envs import step_mdp

data = step_mdp(stepped_data)
print(data)
```

```
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

## Environment rollouts

Writing down those three steps (computing an action, making a step,
moving in the MDP) can be a bit tedious and repetitive. Fortunately,
TorchRL provides a nice [`rollout()`](../reference/generated/torchrl.envs.EnvBase.html#id2) function that
allows you to run them in a closed loop at will:

```
rollout = env.rollout(max_steps=10)
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

This data looks pretty much like the `stepped_data` above with the
exception of its batch-size, which now equates the number of steps we
provided through the `max_steps` argument. The magic of tensordict
doesn't end there: if you're interested in a single transition of this
environment, you can index the tensordict like you would index a tensor:

```
transition = rollout[3]
print(transition)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) will automatically check if the index you
provided is a key (in which case we index along the key-dimension) or a
spatial index like here.

Executed as such (without a policy), the `rollout` method may seem rather
useless: it just runs random actions. If a policy is available, it can
be passed to the method and used to collect data.

Nevertheless, it can useful to run a naive, policyless rollout at first to
check what is to be expected from an environment at a glance.

To appreciate the versatility of TorchRL's API, consider the fact that the
rollout method is universally applicable. It functions across **all** use
cases, whether you're working with a single environment like this one,
multiple copies across various processes, a multi-agent environment, or even
a stateless version of it!

## Transforming an environment

Most of the time, you'll want to modify the output of the environment to
better suit your requirements. For example, you might want to monitor the
number of steps executed since the last reset, resize images, or stack
consecutive observations together.

In this section, we'll examine a simple transform, the
[`StepCounter`](../reference/generated/torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) transform.
The complete list of transforms can be found
[here](../reference/envs_transforms.html#transforms).

The transform is integrated with the environment through a
[`TransformedEnv`](../reference/generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv):

```
from torchrl.envs import StepCounter, TransformedEnv

transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
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
 step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
```

As you can see, our environment now has one more entry, `"step_count"` that
tracks the number of steps since the last reset.
Given that we passed the optional
argument `max_steps=10` to the transform constructor, we also truncated the
trajectory after 10 steps (not completing a full rollout of 100 steps like
we asked with the `rollout` call). We can see that the trajectory was
truncated by looking at the truncated entry:

```
print(rollout["next", "truncated"])
```

```
tensor([[False],
 [False],
 [False],
 [False],
 [False],
 [False],
 [False],
 [False],
 [False],
 [ True]])
```

This is all for this short introduction to TorchRL's environment API!

## Next steps

To explore further what TorchRL's environments can do, go and check:

- The `step_and_maybe_reset()` method that packs
together [`step()`](../reference/generated/torchrl.envs.EnvBase.html#id4),
[`step_mdp()`](../reference/generated/torchrl.envs.step_mdp.html#torchrl.envs.step_mdp) and
[`reset()`](../reference/generated/torchrl.envs.EnvBase.html#id1).
- Some environments like [`GymEnv`](../reference/generated/torchrl.envs.GymEnv.html#torchrl.envs.GymEnv) support rendering
through the `from_pixels` argument. Check the class docstrings to know
more!
- The batched environments, in particular [`ParallelEnv`](../reference/generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv)
which allows you to run multiple copies of one same (or different!)
environments on multiple processes.
- Design your own environment with the
[Pendulum tutorial](pendulum.html#pendulum-tuto) and learn about specs and
stateless environments.
- See the more in-depth tutorial about environments
[in the dedicated tutorial](torchrl_envs.html#envs-tuto);
- Check the
[multi-agent environment API](../reference/envs_multiagent.html#marl-environment-api)
if you're interested in MARL;
- TorchRL has many tools to interact with the Gym API such as
a way to register TorchRL envs in the Gym register through
[`register_gym()`](../reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.register_gym), an API to read
the info dictionaries through
`set_info_dict_reader()` or a way
to control the gym backend thanks to
[`set_gym_backend()`](../reference/generated/torchrl.envs.set_gym_backend.html#torchrl.envs.set_gym_backend).

**Total running time of the script:** (0 minutes 0.031 seconds)

[`Download Jupyter notebook: getting-started-0.ipynb`](../_downloads/fea30f51a5adb1722e9e73b12c230828/getting-started-0.ipynb)

[`Download Python source code: getting-started-0.py`](../_downloads/4a24bc24ff84e230d8db066011348430/getting-started-0.py)

[`Download zipped: getting-started-0.zip`](../_downloads/dad3f0a1f7722e8534bfbd892b0c68bd/getting-started-0.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)