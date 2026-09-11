"""
Deploying low-level policies for high-level reinforcement learning
=================================================================

What you will learn
-------------------

Reuse a policy behind a smaller decision space, adapt its inputs with existing
TensorDict transforms, and train the high-level actor with an ordinary PPO
collector and loss. The same deployment supports shared policies over agents
and explicit recurrent state. This example needs PyTorch, TensorDict, and
TorchRL; it requires no robot assets or external simulator.

The physical task is TorchRL's pendulum. A low-level velocity servo produces
torques. The high-level policy chooses a target velocity every five physical
steps and receives the sum of the task rewards.
"""

from __future__ import annotations

import os

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import Bounded, Composite
from torchrl.envs import PendulumEnv, TransformedEnv
from torchrl.envs.transforms import (
    CatTensors,
    ClosedLoopMultiAction,
    Compose,
    StepCounter,
    UnsqueezeTransform,
)
from torchrl.envs.utils import check_env_specs
from torchrl.modules import LowLevelController, MLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

torch.manual_seed(0)
fast = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"

# %%
# A task and an existing controller
# --------------------------------
#
# These transforms expose a vector observation while preserving the pendulum's
# scalar state inputs. A physical episode lasts at most 200 control steps.

task_env = TransformedEnv(
    PendulumEnv(),
    Compose(
        UnsqueezeTransform(-1, in_keys=["th", "thdot"], out_keys=["angle", "velocity"]),
        CatTensors(
            in_keys=["angle", "velocity"],
            out_key="observation",
            sort=False,
            del_keys=False,
        ),
        StepCounter(max_steps=200),
    ),
)

# This fixed servo keeps the example reproducible without a checkpoint download.
# In an application, supply the already-trained TensorDict policy here, with its
# original input/output keys and recurrent primers. Its observation is velocity
# followed by target velocity. Its output is the physical torque.

servo = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    servo.weight.copy_(torch.tensor([[-0.5, 0.5]]))
low_level_policy = TensorDictModule(
    nn.Sequential(servo, nn.Hardtanh(-2.0, 2.0)),
    in_keys=["observation"],
    out_keys=["action"],
)

# %%
# Deployment is two objects
# -------------------------
#
# The decision spec describes one controller. CatTensors adapts the task's
# TensorDict without overwriting the high-level actor's observation: it runs in
# the controller's private workspace.

controller = LowLevelController(
    low_level_policy,
    decision_spec=Composite(target_velocity=Bounded(-4.0, 4.0, shape=(1,))),
    adapter=CatTensors(
        in_keys=["velocity", "target_velocity"], out_key="observation", sort=False
    ),
)
training_env = ClosedLoopMultiAction.from_env(
    task_env, controller, steps=5, reward_aggregation="sum"
)
check_env_specs(training_env)

# %%
# Ordinary PPO
# -------------
#
# Only the high-level actor goes into the collector. The low-level controller
# belongs to the environment. Low-level inference is deterministic and has
# gradients disabled by default; this context ends before the next actor call.
# Gamma below discounts one high-level decision, not one physical step.

actor = ProbabilisticActor(
    module=TensorDictModule(
        nn.Sequential(
            MLP(in_features=2, out_features=2, num_cells=[64, 64]),
            NormalParamExtractor(),
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    ),
    spec=training_env.full_action_spec_unbatched,
    in_keys=["loc", "scale"],
    out_keys=[training_env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={"low": -4.0, "high": 4.0},
    return_log_prob=True,
)
critic = TensorDictModule(
    MLP(in_features=2, out_features=1, num_cells=[64, 64]),
    in_keys=["observation"],
    out_keys=["state_value"],
)
loss = ClipPPOLoss(actor, critic, normalize_advantage=True, entropy_coeff=0.01)
loss.set_keys(action=training_env.action_key)
loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
optimizer = torch.optim.Adam(loss.parameters(), lr=3e-4)
collector = Collector(
    training_env,
    actor,
    frames_per_batch=128,
    total_frames=256 if fast else 2560,
)
history = []
low_level_weights = [p.detach().clone() for p in low_level_policy.parameters()]
try:
    for batch in collector:
        with torch.no_grad():
            loss.value_estimator(
                batch,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )
        for _ in range(4):
            losses = loss(batch.reshape(-1))
            objective = (
                losses["loss_objective"]
                + losses["loss_critic"]
                + losses["loss_entropy"]
            )
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        history.append(batch["next", "reward"].mean().item())
        collector.update_policy_weights_()
finally:
    collector.shutdown()

for before, after in zip(low_level_weights, low_level_policy.parameters()):
    torch.testing.assert_close(before, after)
    assert after.grad is None

# %%
# Many agents and recurrent policies
# ----------------------------------
#
# For a task with 64 environments and ten agents, add group_key="agents".
# A per-controller decision with shape [2] becomes an environment action spec
# with shape [64, 10, 2]. The policy sees 640 independent rows with shared
# weights. This works for agents, robot effectors, or another explicitly
# batched controller group; no football or team semantics are built in.
#
# .. code-block:: python
#
#    controller = LowLevelController(
#        pretrained_policy,
#        Composite(target_velocity=Bounded(-1, 1, shape=(2,))),
#        adapter=CatTensors(
#            in_keys=["proprioception", "target_velocity"],
#            out_key="observation", sort=False,
#        ),
#        group_key="agents",
#        reset_key="fallen",  # Optional additional per-agent reset signal.
#    )
#    training_env = ClosedLoopMultiAction.from_env(task_env, controller, steps=5)
#
# State declared by policy or adapter primers lives under
# ("agents", "_controller"). The pretrained policy still reads its original
# recurrent_state and is_init keys. Every physical step advances its state,
# including the final step. An individual reset restores that row's declared
# defaults and sets its is_init flag. Hidden mutable Python state cannot be
# replicated across agents; make persistent state explicit in the TensorDict.
#
# The actor and critic read ("agents", "observation"), and the actor writes
# training_env.action_key. Configure the PPO loss's action, reward and value
# keys accordingly. For environment-level episode boundaries, expand done
# and terminated to the per-agent reward shape before computing GAE:
#
# .. code-block:: python
#
#    reward = batch["next", "agents", "reward"]
#    for key in ("done", "terminated"):
#        batch["next", "agents", key] = (
#            batch["next", key].unsqueeze(-1).expand_as(reward)
#        )
#
# Choosing centralized critics, teams, opponent policies, or self-play belongs
# to the high-level training setup. MicroDuck tasks can use the same machinery
# through microduck_skill_env(task_env, walker, tasks, steps=5); the preset only
# converts commands and gait clocks and adds last-skill/fall observations.

# %%
# Configuration
# --------------
#
# LowLevelControllerConfig and ClosedLoopMultiActionConfig expose the same
# constructor arguments. Hydra can also target the public constructors directly:
#
# .. code-block:: yaml
#
#    controller:
#      _target_: torchrl.modules.LowLevelController
#      decision_spec:
#        _target_: torchrl.data.Composite
#        target_velocity:
#          _target_: torchrl.data.Bounded
#          low: -4.0
#          high: 4.0
#          shape: [1]
#      adapter:
#        _target_: torchrl.envs.transforms.CatTensors
#        in_keys: [velocity, target_velocity]
#        out_key: observation
#        sort: false
#    deployment:
#      _target_: torchrl.envs.transforms.ClosedLoopMultiAction.from_env
#      steps: 5
#      reward_aggregation: sum
#
# .. code-block:: python
#
#    controller = instantiate(cfg.controller, policy=low_level_policy)
#    training_env = instantiate(cfg.deployment, env=task_env, controller=controller)
#
# Declare the Composite spec through its Hydra target, as above. OmegaConf
# interprets raw spec objects as dataclasses, so passing a constructed Composite
# as an instantiate override is unsuitable.

# %%
# Conclusion
# -----------
#
# LowLevelController adapts policy interfaces and isolates per-instance state.
# ClosedLoopMultiAction supplies fresh feedback at each physical step, stops
# finished environments, and reduces rewards. Existing collectors and losses
# then train the high-level policy without a second execution framework.
#
# Further reading
# ----------------
#
# See :class:`~torchrl.modules.LowLevelController`,
# :class:`~torchrl.envs.transforms.ClosedLoopMultiAction`,
# :class:`~torchrl.modules.GRUModule`, and the multi-agent PPO tutorial.
