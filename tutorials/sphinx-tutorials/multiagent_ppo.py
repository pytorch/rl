# -*- coding: utf-8 -*-
"""
Multi-Agent Reinforcement Learning (PPO) with TorchRL Tutorial
==================================================
**Author**: `Matteo Bettini <https://github.com/matteobettini>`_

This tutorial demonstrates how to use PyTorch and :py:mod:`torchrl` to
solve a Multi-Agent Reinforcement Learning (MARL) problem.

A code-only version of this tutorial is available in the
`torchrl examples <https://github.com/pytorch/rl/tree/main/examples/multiagent/mappo_ippo.py>`__,
alongside other simple scripts for many MARL algorithms (QMIX, MADDPG, IQL).

For ease of use, this tutorial will follow the general structure of the already available
`single agent PPO tutorial <https://pytorch.org/rl/tutorials/coding_ppo.html>`__.
It is suggested but not mandatory to get familiar with that prior to starting this tutorial.

In this tutorial, we will use the *Navigation* environment from
`VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__,
a multi-robot simulator, also
based on PyTorch, that runs parallel batched simulation on device.

In the *Navigation* environment,
we need to train multiple robots (spawned at random positions)
to navigate to their goals (also at random positions), while
using  `LIDAR sensors <https://en.wikipedia.org/wiki/Lidar>`__ to avoid collisions among each other.

.. figure:: /_static/img/navigation.gif
   :alt: Navigation

   Multi-agent *Navigation* scenario

Key learnings:

- How to create a multi-agent environment in TorchRL, how its specs work, and how it integrates with the library;
- How you use GPU vectorized environments in TorchRL;
- How to create different multi-agent network architectures in TorchRL (e.g., using parameter sharing, centralised critic)
- How we can use :class:`tensordict.TensorDict` to carry multi-agent data;
- How we can tie all the library components (collectors, modules, replay buffers, and losses) in a multi-agent MAPPO/IPPO training loop.

"""

######################################################################
# If you are running this in Google Colab, make sure you install the following dependencies:
#
# .. code-block:: bash
#
#    !pip3 install torchrl
#    !pip3 install vmas
#    !pip3 install tqdm
#
# Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
# batch of data is being collected and directly consumed to train the policy to maximise
# the expected return given some proximality constraints. You can think of it
# as a sophisticated version of `REINFORCE <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`_,
# the foundational policy-optimization algorithm. For more information, see the
# `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_ paper.
#
# This type of algorithms is usually trained *on-policy*. This means that, at every learning iteration, we have a
# **sampling** and a **training** phase. In the **sampling** phase of iteration :math:`t`, rollouts are collected
# form agents' interactions in the environment using the current policies :math:`\mathbf{\pi}_t`.
# In the **training** phase, all the collected rollouts are immediately fed to the training process to perform
# backpropagation. This leads to updated policies which are then used again for sampling.
# The execution of this process in a loop constitutes *on-policy learning*.
#
# .. figure:: /_static/img/on_policy_vmas.png
#    :alt: On-policy learning
#
#    On-policy learning
#
#
# In the training phase of the PPO algorithm, a *critic* is used to estimate the goodness of the actions
# taken by the policy. The critic learns to approximate the value (mean discounted return) of a specific state.
# The PPO loss then compares the actual return obtained by the policy to the one estimated by the critic to determine
# the advantage of the action taken and guide the policy optimization.
#
# In multi-agent settings, things are a bit different. We now have multiple policies :math:`\mathbf{\pi}`,
# one for each agent. Policies are typically local and decentralised. This means that
# the policy for a single agent will output an action for that agent based only on its observation.
# In the MARL literature, this is referred to as **decentralized execution**.
# On the other hand, different formulations exist for the critic, mainly:
#
# - In `MAPPO <https://arxiv.org/abs/2103.01955>`_ the critic is centralised and takes as input the global state
#   of the system. This can be a global observation or simply the concatenation of the agents' observation. MAPPO
#   can be used in contexts where **centralised training** is performed as it needs access to global information.
# - In `IPPO <https://arxiv.org/abs/2011.09533>`_ the critic takes as input just the observation of the respective agent,
#   exactly like the policy. This allows **decentralised training** as both the critic and the policy will only need local
#   information to compute their outputs.
#
# Centralised critics help overcome the non-stationary of multiple agents learning concurrently, but,
# on the other hand, they may be impacted by their large input space.
# In this tutorial, we will be able to train both formulations, and we will also discuss how
# parameter-sharing (the practice of sharing the network parameters across the agents) impacts each.
#
# This tutorial is structured as follows:
#
# 1. First, we will define a set of hyperparameters we will be using.
#
# 2. Next, we will create a vectorized multi-agent environment, using TorchRL's
#    wrapper for the VMAS simulator.
#
# 3. Next, we will design the policy and the critic networks, discussing the impact of the various choices on
#    parameter sharing and critic centralisation.
#
# 4. Next, we will create the sampling collector and the replay buffer.
#
# 5. Finally, we will run our training loop and analyze the results.
#
# If you are running this in Colab or in a machine with a GUI, you will also have the option
# to render and visualize your own trained policy prior and after training.
#
# Let's import our dependencies
#

# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueOperator

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from tqdm import tqdm


######################################################################
# Define Hyperparameters
# ----------------------
#
# We set the hyperparameters for our tutorial.
# Depending on the resources
# available, one may choose to execute the policy and the simulator on GPU or on another
# device.
# You can tune some of these values to adjust the computational requirements.
#

# Devices
device = "cpu" if not torch.has_cuda else "cuda:0"  # The divice where learning is run
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.9  # discount factor
lmbda = 0.9  # lambda for generalized advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

######################################################################
# Environment
# ---------------------
#
# Multi-agent environments simulate multiple agents interacting with the world.
# TorchRL API allows integrating various types of multi-agent environment flavours.
# Some examples include environments with shared or individual agent rewards, done flags, and observations.
# For more information on how the multi-agent environments API works in TorchRL, you can check out the dedicated
# `doc section <https://pytorch.org/rl/reference/envs.html#multi-agent-environments>`_.
#
# The VMAS simulator, in particular, models agents with individual rewards, info, observations, and actions, but
# with a collective done flag.
# Furthermore, it uses *vectorization* to perform simulation in a batch.
# This means that all its state and physics
# are PyTorch tensors with a first dimension representing the number of parallel environments in a batch.
# This allows leveraging the Single Instruction Multiple Data (SIMD) paradigm of GPUs and significantly
# speed up parallel computation by leveraging parallelization in GPU warps. I also means
# that, when using it in TorchRL, both simulation and training can be run on-device, without ever passing
# data to the CPU.
#
# The multi-agent task we will solve today is *Navigation* (see animated figure above).
# In *Navigation*, randomly spawned agents
# (circles with surrounding dots) need to navigate
# to randomly spawned goals (smaller circles).
# Agents need to use LIDARs (dots around them) to
# avoid colliding into each other.
# Agents act in a 2D continuous world with drag and elastic collisions.
# Their actions are 2D continuous forces which determine their acceleration.
# The reward is composed of three terms: a collision penalization, a reward based on the distance to the goal, and a
# final shared reward given when all agents reach their goal.
# The distance-based term is computed as the difference in the relative distance
# between an agent and its goal over two consecutive timesteps.
# Each agent observes its position,
# velocity, lidar readings, and relative position to its goal.
#
# We will now instantiate the environment.
# For this tutorial, we will limit the episodes to ``max_steps``, after which the done flag is set. This is
# functionality is already provided in the VMAS simulator but the TorchRL ``StepCount``
# transform could alternatively be used.
# We will also use ``num_vmas_envs`` vectorized environments, to leverage batch simulation.
#
#

max_steps = 100  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "navigation"
n_agents = 3

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
)

######################################################################
# The environment is not only defined by its simulator and transforms, but also
# by a series of metadata that describe what can be expected during its
# execution.
# For efficiency purposes, TorchRL is quite stringent when it comes to
# environment specs, but you can easily check that your environment specs are
# adequate.
# In our example, the :class:`VmasEnv` takes care of setting the proper specs for your env so
# you should not have to care about this.
#
# There are four specs to look at:
#
# - ``action_spec`` defines the action space;
# - ``reward_spec`` defines the reward domain;
# - ``done_spec`` defines the done domain;
# - ``observation_spec`` which defines the domain of all other outputs from environmnet steps;
#
#

print("action_spec:", env.action_spec)
print("reward_spec:", env.reward_spec)
print("done_spec:", env.done_spec)
print("observation_spec:", env.observation_spec)


######################################################################
# Using the commands just shown we can access the domain of each value.
# Doing this we can see that all specs apart from done have a leading shape ``(num_vmas_envs, n_agents)``.
# This represents the fact that those values will be present for each agent in each individual environment.
# The done spec, on the other hand, has leading shape ``(num_vmas_envs)``, representing that done is shared among
# agents.
#
# TorchRL has a way to keep track of which MARL specs are shared and which are not.
# In fact, specs that have the additional agent dimension
# (i.e., they vary for each agent) will be contained in a inner "agents" key.
#
# To access the full structure of the specs we can use
#

print("full_action_spec:", env.input_spec["_action_spec"])
print("full_reward_spec:", env.output_spec["_reward_spec"])
print("full_done_spec:", env.output_spec["_done_spec"])

######################################################################
# As you can see the reward and action spec present the "agent" key,
# meaning that entries in tensordicts belonging to those specs will be nested in an "agents" tensordict,
# grouping all per-agent values.
#
# To quickly access the key for each of these values in tensordicts, we can simply ask the environment for the
# respective key, and
# we will immediately understand which are per-agent and which shared.
# This info will be useful in order to tell all other TorchRL components where to find each value
#

print("action_key:", env.action_key)
print("reward_key:", env.reward_key)
print("done_key:", env.done_key)

######################################################################
# Transforms
# ~~~~~~~~~~
#
# We can append any torchrl transform we need to our enviornment.
# These will modify its input output in some desired way.
# Remember that, in multi-agent contexts, it is paramount to provide explicitly the keys to modify.
#
# For example, in this case we will instantiate a ``RewardSum`` transform which will sum rewards over the episode.
# We will tell this transform where to find the reward key and where to write the summed episode reward.
# The transformed env will inherit
# the device and meta-data of the wrapped env, and transform these depending on the sequence
# of transforms it contains.
#


env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)


######################################################################
# the :func:`check_env_specs` function runs a small rollout and compares its output against the environment
# specs. If no error is raised, we can be confident that the specs are properly defined:
#
check_env_specs(env)

######################################################################
# For fun, let's see what a simple random rollout looks like. You can
# call `env.rollout(n_steps)` and get an overview of what the environment inputs
# and outputs look like. Actions will automatically be drawn from the action spec
# domain, so you don't need to care about designing a random sampler.
#
# Typically, at each step, an RL environment receives an
# action as input, and outputs an observation, a reward and a done state. The
# observation may be composite, meaning that it could be composed of more than one
# tensor. This is not a problem for TorchRL, since the whole set of observations
# is automatically packed in the output :class:`tensordict.TensorDict`. After executing a rollout
# (ie a sequence of environment steps and random action generations) over a given
# number of steps, we will retrieve a :class:`tensordict.TensorDict` instance with a shape
# that matches this trajectory length:
#
rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

######################################################################
# Our rollout data has a shape of ``torch.Size([3])``, which matches the number of steps
# we ran it for. The ``"next"`` entry points to the data coming after the current step.
# In most cases, the ``"next""`` data at time `t` matches the data at ``t+1``, but this
# may not be the case if we are using some specific transformations (e.g. multi-step).
#
# Policy
# ------
#
# PPO utilizes a stochastic policy to handle exploration. This means that our
# neural network will have to output the parameters of a distribution, rather
# than a single value corresponding to the action taken.
#
# As the data is continuous, we use a Tanh-Normal distribution to respect the
# action space boundaries. TorchRL provides such distribution, and the only
# thing we need to care about is to build a neural network that outputs the
# right number of parameters for the policy to work with (a location, or mean,
# and a scale):
#
# .. math::
#
#     f_{\theta}(\text{observation}) = \mu_{\theta}(\text{observation}), \sigma^{+}_{\theta}(\text{observation})
#
# The only extra-difficulty that is brought up here is to split our output in two
# equal parts and map the second to a scrictly positive space.
#
# We design the policy in three steps:
#
# 1. Define a neural network ``D_obs`` -> ``2 * D_action``. Indeed, our ``loc`` (mu) and ``scale`` (sigma) both have dimension ``D_action``.
#
# 2. Append a :class:`NormalParamExtractor` to extract a location and a scale (ie splits the input in two equal parts
#    and applies a positive transformation to the scale parameter).
#
# 3. Create a probabilistic :class:`TensorDictModule` that can generate this distribution and sample from it.
#
#
shared_parameters_policy = True

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=shared_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),
)
######################################################################
# To enable the policy to "talk" with the environment through the tensordict
# data carrier, we wrap the ``nn.Module`` in a :class:`TensorDictModule`. This
# class will simply ready the ``in_keys`` it is provided with and write the
# outputs in-place at the registered ``out_keys``.
#
policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

######################################################################
# We now need to build a distribution out of the location and scale of our
# normal distribution. To do so, we instruct the :class:`ProbabilisticActor`
# class to build a :class:`TanhNormal` out of the location and scale
# parameters. We also provide the minimum and maximum values of this
# distribution, which we gather from the environment specs.
#
# The name of the ``in_keys`` (and hence the name of the ``out_keys`` from
# the :class:`TensorDictModule` above) cannot be set to any value one may
# like, as the :class:`TanhNormal` distribution constructor will expect the
# ``loc`` and ``scale`` keyword arguments. That being said,
# :class:`ProbabilisticActor` also accepts ``Dict[str, str]`` typed ``in_keys``
# where the key-value pair indicates what ``in_key`` string should be used for
# every keyword argument that is to be used.
#

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.unbatched_action_spec[("agents", "action")].space.minimum,
        "max": env.unbatched_action_spec[("agents", "action")].space.maximum,
    },
    return_log_prob=True,
)  # we'll need the log-prob for the numerator of the importance weights


######################################################################
# Value network
# -------------
#
# The value network is a crucial component of the PPO algorithm, even though it
# won't be used at inference time. This module will read the observations and
# return an estimation of the discounted return for the following trajectory.
# This allows us to amortize learning by relying on the some utility estimation
# that is learnt on-the-fly during training. Our value network share the same
# structure as the policy, but for simplicity we assign it its own set of
# parameters.
#

shared_parameters_critic = True
mappo = True
critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=shared_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic_module = ValueOperator(
    module=critic_net,
    in_keys=[("agents", "observation")],
)

######################################################################
# let's try our policy and value modules. As we said earlier, the usage of
# :class:`TensorDictModule` makes it possible to directly read the output
# of the environment to run these modules, as they know what information to read
# and where to write it:
#
print("Running policy:", policy_module(env.reset()))
print("Running value:", critic_module(env.reset()))

######################################################################
# Data collector
# --------------
#
# TorchRL provides a set of :class:`DataCollector` classes. Briefly, these
# classes execute three operations: reset an environment, compute an action
# given the latest observation, execute a step in the environment, and repeat
# the last two steps until the environment signals a stop (or reaches a done
# state).
#
# They allow you to control how many frames to collect at each iteration
# (through the ``frames_per_batch`` parameter),
# when to reset the environment (through the ``max_frames_per_traj`` argument),
# on which ``device`` the policy should be executed, etc. They are also
# designed to work efficiently with batched and multiprocessed environments.
#
# The simplest data collector is the :class:`SyncDataCollector`: it is an
# iterator that you can use to get batches of data of a given length, and
# that will stop once a total number of frames (``total_frames``) have been
# collected.
# Other data collectors (``MultiSyncDataCollector`` and
# ``MultiaSyncDataCollector``) will execute the same operations in synchronous
# and asynchronous manner over a set of multiprocessed workers.
#
# As for the policy and environment before, the data collector will return
# :class:`tensordict.TensorDict` instances with a total number of elements that will
# match ``frames_per_batch``. Using :class:`tensordict.TensorDict` to pass data to the
# training loop allows you to write dataloading pipelines
# that are 100% oblivious to the actual specificities of the rollout content.
#
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

######################################################################
# Replay buffer
# -------------
#
# Replay buffers are a common building piece of off-policy RL algorithms.
# In on-policy contexts, a replay buffer is refilled every time a batch of
# data is collected, and its data is repeatedly consumed for a certain number
# of epochs.
#
# TorchRL's replay buffers are built using a common container
# :class:`ReplayBuffer` which takes as argument the components of the buffer:
# a storage, a writer, a sampler and possibly some transforms. Only the
# storage (which indicates the replay buffer capacity) is mandatory. We
# also specify a sampler without repetition to avoid sampling multiple times
# the same item in one epoch.
# Using a replay buffer for PPO is not mandatory and we could simply
# sample the sub-batches from the collected batch, but using these classes
# make it easy for us to build the inner training loop in a reproducible way.
#

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

######################################################################
# Loss function
# -------------
#
# The PPO loss can be directly imported from torchrl for convenience using the
# :class:`ClipPPOLoss` class. This is the easiest way of utilizing PPO:
# it hides away the mathematical operations of PPO and the control flow that
# goes with it.
#
# PPO requires some "advantage estimation" to be computed. In short, an advantage
# is a value that reflects an expectancy over the return value while dealing with
# the bias / variance tradeoff.
# To compute the advantage, one just needs to (1) build the advantage module, which
# utilizes our value operator, and (2) pass each batch of data through it before each
# epoch.
# The GAE module will update the input :class:`TensorDict` with new ``"advantage"`` and
# ``"value_target"`` entries.
# The ``"value_target"`` is a gradient-free tensor that represents the empirical
# value that the value network should represent with the input observation.
# Both of these will be used by :class:`ClipPPOLoss` to
# return the policy and value losses.
#

loss_module = ClipPPOLoss(
    actor=policy,
    critic=critic_module,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
)
loss_module.set_keys(reward=env.reward_key, action=env.action_key)
loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
optim = torch.optim.Adam(loss_module.parameters(), lr)

######################################################################
# Training loop
# -------------
# We now have all the pieces needed to code our training loop.
# The steps include:
#
# * Collect data
#
#   * Compute advantage
#
#     * Loop over the collected to compute loss values
#     * Back propagate
#     * Optimize
#     * Repeat
#
#   * Repeat
#
# * Repeat
#

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

total_frames = 0
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get(("next", env.reward_key)).shape),
    )  # We need to expand the done to match the reward shape

    with torch.no_grad():
        loss_module.value_estimator(
            tensordict_data,
            params=loss_module.critic_params.detach(),
            target_params=loss_module.target_critic_params,
        )

    current_frames = tensordict_data.numel()
    total_frames += current_frames
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    training_tds = []
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)
            training_tds.append(loss_vals.detach())

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    done = tensordict_data.get(("next", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )

    training_tds = torch.stack(training_tds)

    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

######################################################################
# Results
# -------
#
# Before the 1M step cap is reached, the algorithm should have reached a max
# step count of 1000 steps, which is the maximum number of steps before the
# trajectory is truncated.
#


######################################################################
# Render
# -------
# If you are running this in a machine with GUI, you can render the trained policy by running:
#
# .. code-block:: python
#
#    env.rollout(
#        max_steps=max_steps,
#        policy=policy,
#        callback=lambda env, _: env.render(),
#        auto_cast_to_device=True,
#        break_when_any_done=False,
#    )
#
# If you are running this in Google Colab, you can render the trained policy by running:
#
# .. code-block:: python
#
#    !apt-get update
#    !apt-get install -y x11-utils
#    !apt-get install -y xvfb
#    !pip install pyvirtualdisplay
#    import pyvirtualdisplay
#    display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
#    display.start()
#    from PIL import Image
#
#    def rendering_callback(env, td):
#        env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
#    env.frames = []
#    env.rollout(
#         max_steps=max_steps,
#         policy=policy,
#         callback=rendering_callback,
#         auto_cast_to_device=True,
#         break_when_any_done=False,
#     )
#     env.frames[0].save(
#        f"{scenario_name}.gif",
#        save_all=True,
#        append_images=env.frames[1:],
#       duration=3,
#       loop=0,
#     )
#
#     from IPython.display import Image
#     Image(open(f"{scenario_name}.gif", "rb").read())
#


######################################################################
# Conclusion and next steps
# -------------------------
#
# In this tutorial, we have learned:
#
# 1. How to create and customize an environment with :py:mod:`torchrl`;
# 2. How to write a model and a loss function;
# 3. How to set up a typical training loop.
#
# If you want to experiment with this tutorial a bit more, you can apply the following modifications:
#
# * From an efficiency perspective,
#   we could run several simulations in parallel to speed up data collection.
#   Check :class:`~torchrl.envs.ParallelEnv` for further information.
#
# * From a logging perspective, one could add a :class:`~torchrl.record.VideoRecorder` transform to
#   the environment after asking for rendering to get a visual rendering of the
#   inverted pendulum in action. Check :py:mod:`torchrl.record` to
#   know more.
#
