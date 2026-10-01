"""
Multi-Agent Reinforcement Learning (COMA) with TorchRL Tutorial
================================================================
**Author**: `The TorchRL team <https://github.com/pytorch/rl>`_

.. seealso::
   This tutorial builds on :doc:`/tutorials/multiagent_ppo`. It is
   suggested but not mandatory to get familiar with that one first, since we
   will skip over the multi-agent environment basics already covered there.

This tutorial demonstrates how to use PyTorch and :py:mod:`torchrl` to train
`COMA <https://arxiv.org/abs/1705.08926>`_ (Counterfactual Multi-Agent
Policy Gradients), a MARL algorithm that tackles the **multi-agent credit
assignment problem**: when the team only receives a single shared reward,
how should each agent know whether its own action helped or hurt?

COMA answers this with a *counterfactual baseline*: a centralised critic
outputs one Q-value per possible action of the acting agent, and each
agent's advantage is obtained by comparing the Q-value of the action it
actually took against the *expected* Q-value it would have gotten had it
acted according to its current policy instead, while every other agent's
action is held fixed. This isolates the part of the outcome that agent's
own choice is responsible for.

In this tutorial, we will use the *Balance* environment from
`VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__, with
its **discrete** action space (COMA, like the original paper, is designed
for discrete actions).

Key learnings:

- How to build a decentralised actor paired with a centralised, per-action
  critic for a discrete-action MARL problem;
- How the counterfactual baseline is computed, and why it addresses credit
  assignment;
- How :meth:`~torchrl.objectives.multiagent.COMALoss.compute_value_target`
  bootstraps an n-step Q-value target along the rollout's time dimension,
  and how it flags rollout-boundary transitions that have no reliable
  bootstrap so they are excluded from the loss automatically;
- How to tie all of this into a full COMA training loop.

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
# Like PPO, COMA is trained *on-policy*: at every iteration we collect a
# batch of rollouts with the current policies, then immediately consume
# them for a few epochs of gradient updates before collecting again.
#
# What is specific to COMA is the *critic*. Rather than estimating a state
# value :math:`V(s)`, the critic estimates, for the acting agent :math:`i`,
# one Q-value per possible action:
# :math:`Q(s, u^{-i}, u^i)` for every :math:`u^i` in the action space, where
# :math:`u^{-i}` are the actions of every other agent. The critic is
# *centralised* in the sense that it conditions on information beyond agent
# :math:`i`'s own observation (typically the other agents' actions or a
# global state), but it is trained once per agent, not once per team, and
# it outputs a value *per action* rather than a single scalar.
#
# Given this critic, the counterfactual advantage for agent :math:`i` is:
#
# .. math::
#
#    A^i(s, u) = Q(s, u^{-i}, u^i) - \sum_{u'^i} \pi^i(u'^i \mid s) \, Q(s, u^{-i}, u'^i)
#
# The first term is the Q-value of the action agent :math:`i` actually
# took. The second term marginalises the critic's own output over agent
# :math:`i`'s action under its current policy, holding every other agent's
# action fixed -- this is the counterfactual baseline. Because it is
# computed from the *same* critic call, no extra network evaluation is
# needed, and because it is specific to agent :math:`i`, it isolates
# exactly the part of the team's Q-value that agent :math:`i`'s own choice
# affected.
#
# This tutorial is structured as follows:
#
# 1. We define a set of hyperparameters.
#
# 2. We create a vectorized, discrete-action multi-agent environment using
#    TorchRL's wrapper for the VMAS simulator, and a helper transform that
#    prepares the critic's other-agents-actions input.
#
# 3. We design the decentralised actor and the centralised, per-action
#    critic.
#
# 4. We create the sampling collector and the replay buffer.
#
# 5. We run our training loop and analyse the results.
#
# Let's import our dependencies
#

# Torch
import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from torch import multiprocessing

# Data collection
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, OneHotCategorical, ProbabilisticActor

# Loss
from torchrl.objectives import SoftUpdate
from torchrl.objectives.multiagent.coma import add_action_without_self, COMALoss

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

######################################################################
# Define Hyperparameters
# ----------------------
#
# We set the hyperparameters for our tutorial. Depending on the resources
# available, one may choose to execute the policy and the simulator on GPU
# or on another device. You can tune some of these values to adjust the
# computational requirements.
#

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 5e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# COMA
gamma = 0.9  # discount factor
n_step = 1  # number of Bellman backups applied by compute_value_target (TD(0) if 1)
qvalue_loss_coef = 0.5  # weight of the critic's MSE loss relative to the actor loss
entropy_eps = 1e-3  # coefficient of the entropy term in the COMA loss
tau = 0.005  # soft-update rate for the target critic

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

######################################################################
# Environment
# -----------
#
# We use the *Balance* VMAS scenario with its **discrete** action space
# (``continuous_actions=False``): each agent picks one out of a small set of
# discrete moves at every step. COMA -- like the original paper, which
# targets StarCraft Multi-Agent Challenge -- assumes a discrete action
# space, since the critic must enumerate every possible action of the
# acting agent.
#

max_steps = 100  # Episode steps before done
num_vmas_envs = frames_per_batch // max_steps
scenario_name = "balance"
n_agents = 3

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=False,
    # COMALoss and its helpers (add_action_without_self, ...) operate on
    # one-hot actions, so we need a OneHot action spec rather than VmasEnv's
    # default Categorical (index-encoded) one.
    categorical_actions=False,
    max_steps=max_steps,
    device=vmas_device,
    n_agents=n_agents,
)
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)
check_env_specs(env)

n_actions = env.full_action_spec[env.action_key].space.n
print("number of discrete actions per agent:", n_actions)

######################################################################
# ``add_action_without_self``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# COMA's critic is centralised through its *inputs*: alongside its own
# observation, each agent's critic call also sees the *other* agents'
# actions. :func:`~torchrl.objectives.multiagent.coma.add_action_without_self`
# is a small tensordict transform that, given a joint one-hot action tensor
# of shape ``(*B, n_agents, n_actions)``, writes for every agent the
# flattened actions of every *other* agent under ``("agents",
# "action_without_self")``, of shape
# ``(*B, n_agents, (n_agents - 1) * n_actions)``.
#
# Let's see it in action on a random rollout:
#
rollout = env.rollout(3)
add_action_without_self(rollout)
print(
    "action shape:",
    rollout["agents", "action"].shape,
    "-> action_without_self shape:",
    rollout["agents", "action_without_self"].shape,
)

######################################################################
# Policy
# ------
#
# The actor is **decentralised**: each agent's policy outputs a categorical
# distribution over its ``n_actions`` discrete moves, conditioned only on
# its own observation.
#

share_parameters_policy = True

actor_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=n_actions,
    n_agents=env.n_agents,
    centralised=False,  # decentralised: each agent acts from its own observation
    share_params=share_parameters_policy,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)
policy_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
)
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.full_action_spec_unbatched,
    in_keys=[("agents", "logits")],
    out_keys=[env.action_key],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

######################################################################
# Critic network
# --------------
#
# The critic conditions on the acting agent's own observation *and* the
# other agents' actions (via ``action_without_self``), and outputs one
# Q-value per possible action of the acting agent. Note that, unlike
# MAPPO's critic, this is not "centralised" through a shared network across
# agents -- :class:`~torchrl.modules.MultiAgentMLP` with ``centralised=False``
# and ``share_params=True`` still gives every agent the same *weights*, but
# each agent's Q-values are computed from its own observation plus the
# other agents' actions, not from the concatenation of everyone's
# observation.
#

share_parameters_critic = True

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
    + (env.n_agents - 1) * n_actions,
    n_agent_outputs=n_actions,  # one Q-value per action of the acting agent
    n_agents=env.n_agents,
    centralised=False,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)
qvalue_module = TensorDictModule(
    critic_net,
    in_keys=[("agents", "observation"), ("agents", "action_without_self")],
    out_keys=[("agents", "action_value")],
)

######################################################################
# Let's try our policy and critic on the rollout we collected above (after
# adding the ``action_without_self`` key, which the critic needs):
#
print("Running policy:", policy(env.reset()))
print("Running qvalue:", qvalue_module(rollout))

######################################################################
# Data collector and replay buffer
# ---------------------------------
#
# These are the same building blocks used across TorchRL's on-policy
# algorithms; see :doc:`/tutorials/multiagent_ppo` for a more detailed
# walkthrough.
#
collector = Collector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

######################################################################
# Loss function
# -------------
#
# :class:`~torchrl.objectives.multiagent.COMALoss` needs both networks:
# the actor to compute the log-probability and the counterfactual baseline,
# and the critic to compute the chosen-action Q-value.
#
# A target critic is created automatically (COMA bootstraps its Q-value
# target from a target network, just like DQN), so we also need a target
# updater.
#
loss_module = COMALoss(
    actor_network=policy,
    qvalue_network=qvalue_module,
    gamma=gamma,
    qvalue_loss_coef=qvalue_loss_coef,
    entropy_coef=entropy_eps,
    n_step=n_step,
)
loss_module.set_keys(action=env.action_key)
target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

optim = torch.optim.Adam(loss_module.parameters(), lr)

######################################################################
# Training loop
# -------------
#
# The critical difference with a GAE-based loop (as in the PPO tutorial) is
# :meth:`~torchrl.objectives.multiagent.COMALoss.compute_value_target`: it
# must be called on the *freshly collected* rollout, before it is flattened
# into the replay buffer, because it bootstraps the Q-value target along
# the rollout's time dimension (``G_k(t) = r(t) + \gamma (1 - terminated(t))
# \, G_{k-1}(t+1)``, applied ``n_step`` times). It also writes a
# ``"shifted_valid"`` mask: transitions at the very end of a rollout window
# that has not actually terminated have no real "next" data to bootstrap
# from, so rather than silently biasing their target towards zero, they are
# flagged invalid and automatically excluded from the loss (through
# :meth:`~torchrl.objectives.common.LossModule._reduce_loss`).
#
# The steps are:
#
# * Collect data
#     * Add ``action_without_self`` and compute the Q-value target
#         * Loop over epochs
#             * Loop over minibatches to compute loss values
#                 * Back propagate
#                 * Optimise
#                 * Soft-update the target critic
#             * Repeat
#         * Repeat
#     * Repeat
# * Repeat
#

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    # VMAS reports a single team-shared done/terminated at the root; COMALoss
    # expects a per-agent one, so we broadcast it onto the agent dimension.
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    add_action_without_self(tensordict_data)
    with torch.no_grad():
        loss_module.compute_value_target(tensordict_data)

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_actor"]
                + loss_vals["loss_qvalue"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()
            target_net_updater.step()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

######################################################################
# Results
# -------
#
# Let's plot the mean reward obtained per episode.
#
# To make training last longer, increase the ``n_iters`` hyperparameter.
#
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.show()

######################################################################
# Render
# ------
#
# If you are running this in a machine with GUI, you can render the trained policy by running:
#
# .. code-block:: python
#
#    with torch.no_grad():
#       env.rollout(
#           max_steps=max_steps,
#           policy=policy,
#           callback=lambda env, _: env.render(),
#           auto_cast_to_device=True,
#           break_when_any_done=False,
#       )
#

######################################################################
# Conclusion and next steps
# --------------------------
#
# In this tutorial, we have seen:
#
# - How to build a decentralised actor and a centralised, per-action critic
#   for a discrete-action MARL problem, using
#   :func:`~torchrl.objectives.multiagent.coma.add_action_without_self` to
#   feed the critic the other agents' actions;
# - How the counterfactual baseline addresses multi-agent credit assignment
#   without needing to factorise the joint reward;
# - How :meth:`~torchrl.objectives.multiagent.COMALoss.compute_value_target`
#   bootstraps along the rollout's time dimension and masks out rollout
#   boundaries with no reliable bootstrap value;
# - How to tie all of this into a full COMA training loop.
#
# You can check out all the TorchRL multi-agent implementations
# (including COMA, MAPPO/IPPO, QMIX/VDN, MADDPG/IDDPG, and more) as
# code-only Hydra scripts under ``sota-implementations/multiagent`` in the
# GitHub repository.
#
# If you want to compare COMA against a state-value critic instead of a
# per-action one, check out :doc:`/tutorials/multiagent_ppo` (MAPPO/IPPO)
# and :class:`~torchrl.objectives.multiagent.QMixerLoss` (QMix/VDN).
#
