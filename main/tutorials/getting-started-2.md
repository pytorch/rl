Note

Go to the end
to download the full example code.

# Getting started with model optimization

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

In TorchRL, we try to treat optimization as it is custom to do in PyTorch,
using dedicated loss modules which are designed with the sole purpose of
optimizing the model. This approach efficiently decouples the execution of
the policy from its training and allows us to design training loops that are
similar to what can be found in traditional supervised learning examples.

The typical training loop therefore looks like this:

> ```
> >>> for i in range(n_collections):
> ... data = get_next_batch(env, policy)
> ... for j in range(n_optim):
> ... loss = loss_fn(data)
> ... loss.backward()
> ... optim.step()
> ```

In this concise tutorial, you will receive a brief overview of the loss modules. Due to the typically
straightforward nature of the API for basic usage, this tutorial will be kept brief.

## RL objective functions

In RL, innovation typically involves the exploration of novel methods
for optimizing a policy (i.e., new algorithms), rather than focusing
on new architectures, as seen in other domains. Within TorchRL,
these algorithms are encapsulated within loss modules. A loss
module orchestrates the various components of your algorithm and
yields a set of loss values that can be backpropagated
through to train the corresponding components.

In this tutorial, we will take a popular
off-policy algorithm as an example,
[DDPG](https://arxiv.org/abs/1509.02971).

To build a loss module, the only thing one needs is a set of networks
defined as :class:`~tensordict.nn.TensorDictModule`s. Most of the time, one
of these modules will be the policy. Other auxiliary networks such as
Q-Value networks or critics of some kind may be needed as well. Let's see
what this looks like in practice: DDPG requires a deterministic
map from the observation space to the action space as well as a value
network that predicts the value of a state-action pair. The DDPG loss will
attempt to find the policy parameters that output actions that maximize the
value for a given state.

To build the loss, we need both the actor and value networks.
If they are built according to DDPG's expectations, it is all
we need to get a trainable loss module:

```
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

from torchrl.modules import Actor, MLP, ValueOperator
from torchrl.objectives import DDPGLoss

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(
 MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
 in_keys=["observation", "action"],
)

ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)
```

And that is it! Our loss module can now be run with data coming from the
environment (we omit exploration, storage and other features to focus on
the loss functionality):

```
rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)
print(loss_vals)
```

```
TensorDict(
 fields={
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 td_error: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

## LossModule's output

As you can see, the value we received from the loss isn't a single scalar
but a dictionary containing multiple losses.

The reason is simple: because more than one network may be trained at a time,
and since some users may wish to separate the optimization of each module
in distinct steps, TorchRL's objectives will return dictionaries containing
the various loss components.

This format also allows us to pass metadata along with the loss values. In
general, we make sure that only the loss values are differentiable such that
you can simply sum over the values of the dictionary to obtain the total
loss. If you want to make sure you're fully in control of what is happening,
you can sum over only the entries which keys start with the `"loss_"` prefix:

```
total_loss = 0
for key, val in loss_vals.items():
 if key.startswith("loss_"):
 total_loss += val
```

## Training a LossModule

Given all this, training the modules is not so different from what would be
done in any other training loop. Because it wraps the modules,
the easiest way to get the list of trainable parameters is to query
the [`parameters()`](../reference/generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.parameters) method.

We'll need an optimizer (or one optimizer
per module if that is your choice).

```
from torch.optim import Adam

optim = Adam(ddpg_loss.parameters())
total_loss.backward()
```

The following items will typically be
found in your training loop:

```
optim.step()
optim.zero_grad()
```

## Execution modes and context managers

RL training loops often combine several independent controls. Keeping them
separate helps avoid subtle bugs:

- `module.train()` and `module.eval()` are regular PyTorch module modes.
They control modules such as [`Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout),
`BatchNorm`, and
[`ConsistentDropoutModule`](../reference/generated/torchrl.modules.ConsistentDropoutModule.html#torchrl.modules.ConsistentDropoutModule). They do **not** enable or
disable gradients, and they do not decide whether a probabilistic policy
samples randomly or deterministically. For PPO-style losses, it is usually
recommended to keep the policy and value networks in `eval` mode during
both data collection and optimization so that replayed log-probabilities are
computed under the same module behavior as the rollout.
- `ExplorationType` and
`set_exploration_type()` control the *interaction*
mode of TorchRL probabilistic and exploration modules. `RANDOM` samples
from the policy or adds exploration noise, while `MODE`, `MEAN` or
`DETERMINISTIC` select deterministic actions when available. TorchRL's
`ExplorationType` is an alias of TensorDict's `InteractionType`: this
switch is about action selection, not about PyTorch train/eval mode or
autograd. Collectors use this mode during data collection, and loss modules
use their `deterministic_sampling_mode` when re-evaluating policies for
objective computation.
- `no_grad()` and `inference_mode()` control autograd
bookkeeping. They reduce memory usage and prevent graph construction during
rollouts or evaluation, but they do **not** call `eval()` and they do not
change the exploration/interaction mode. `inference_mode` is more
restrictive than `no_grad` and is best reserved for pure inference where
tensors produced in the block will not later be used in autograd-aware code.
- [`set_recurrent_mode`](../reference/generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) controls how TorchRL recurrent
modules process data: step-by-step sequential execution during collection or
full time-batched execution during losses and advantage computation. It does
not affect dropout, batch normalization, gradients, or action sampling.
TorchRL loss modules enter recurrent mode by default when replaying batches.
See [Recurrent training on sequence batches](recurrent_sequence_training.html#recurrent-sequence-tuto)
for a worked example of the collector/loss interaction.

TorchRL collectors set an exploration/interaction mode for collection and run
the policy under `torch.no_grad()`, but they do **not** switch the policy to
`eval` mode. Set the policy mode before passing it to a collector, especially
when the collector may wrap, move, or copy the policy.

There is one important exception to the blanket "keep it in `eval`" rule:
if you intentionally use dropout as the exploration mechanism, prefer
[`ConsistentDropoutModule`](../reference/generated/torchrl.modules.ConsistentDropoutModule.html#torchrl.modules.ConsistentDropoutModule) and manage its train/eval
mode deliberately. This module stores and reuses trajectory-level dropout masks
in train mode; putting the whole policy in `eval` mode disables that
dropout, just like regular PyTorch dropout.

## Further considerations: Target parameters

Another important aspect to consider is the presence of target parameters
in off-policy algorithms like DDPG. Target parameters typically represent
a delayed or smoothed version of the parameters over time, and they play
a crucial role in value estimation during policy training. Utilizing target
parameters for policy training often proves to be significantly more
efficient compared to using the current configuration of value network
parameters. Generally, managing target parameters is handled by the loss
module, relieving users of direct concern. However, it remains the user's
responsibility to update these values as necessary based on specific
requirements. TorchRL offers a couple of updaters, namely
`HardUpdate` and
`SoftUpdate`,
which can be easily instantiated without requiring in-depth
knowledge of the underlying mechanisms of the loss module.

```
from torchrl.objectives import SoftUpdate

updater = SoftUpdate(ddpg_loss, eps=0.99)
```

In your training loop, you will need to update the target parameters at each
optimization step or each collection step:

```
updater.step()
```

This is all you need to know about loss modules to get started!

To further explore the topic, have a look at:

- The [loss module reference page](../reference/objectives.html#ref-objectives);
- The [Coding a DDPG loss tutorial](coding_ddpg.html#coding-ddpg);
- Losses in action in [PPO](coding_ppo.html#coding-ppo) or [DQN](coding_dqn.html#coding-dqn).

**Total running time of the script:** (0 minutes 0.062 seconds)

[`Download Jupyter notebook: getting-started-2.ipynb`](../_downloads/ae2b99ca949badb94260561dacc8c210/getting-started-2.ipynb)

[`Download Python source code: getting-started-2.py`](../_downloads/7952891385703e05c13ed49e6caeed87/getting-started-2.py)

[`Download zipped: getting-started-2.zip`](../_downloads/53e0992e72aa1cea24016933d54c8450/getting-started-2.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)