# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer policies: inputs, memory and training windows
=========================================================

.. _transformer_policies_tutorial:

What you will learn
-------------------

* What a :class:`~torchrl.modules.TransformerModule` reads and writes.
* How GTrXL memory travels through an environment and a collector.
* How dense replay stores one initial carry alongside a whole training window.
* How to recompute features and perform a masked PPO update.

Requires PyTorch, TorchRL, TensorDict and Gymnasium. The configurable training
script in ``sota-implementations/gtrxl`` hides CartPole's velocity observations;
this short tutorial uses its full observation to focus on the data contract.
"""
from __future__ import annotations

import functools as ft

import torch
from tensordict import TensorDict, TypedTensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv, InitTracker, SerialEnv, TransformedEnv
from torchrl.modules import (
    GTrXL,
    ProbabilisticActor,
    set_recurrent_mode,
    TransformerModule,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

######################################################################
# Two state contracts, one wrapper
# --------------------------------
#
# ``TransformerModule`` selects collection or training with
# ``set_recurrent_mode(False/True)``, just like GRU/LSTM modules. It returns
# **features**, so attach action and value heads to build an RL policy.
#
# The default ``CausalTransformer`` owns a key/value cache internally. Its inputs
# are observation/encoder features and boolean ``is_init``. Its output is one
# feature tensor. It needs ``max_seq_len`` to cover the episode and training
# windows must include the episode start. Cache contents are not in replay and
# weight updates invalidate the cache. That API continues to work unchanged.
#
# ``GTrXL`` instead exposes **caller-owned layer memory**. It uses relative
# attention, reordered layer normalization and GRU-style gates. Each query sees
# at most ``M`` preceding positions plus itself, even in a long training window.
# The memory contains each layer's *inputs*, not projected attention keys/values.
# ``valid`` can represent empty or partially filled memory and masks invalid slots.
#
# Here is the single-step contract, where ``B`` is the number of environments,
# ``F`` the encoder input width, ``D`` the residual width and ``L`` the layers:
#
# .. list-table:: One environment step (TensorDict batch size ``[B]``)
#    :header-rows: 1
#
#    * - Key
#      - Shape / type
#      - Meaning
#    * - ``observation``
#      - ``[B, F]``, float
#      - Current observation or upstream encoder output.
#    * - ``state.memory``
#      - ``[B, L, M, D]``, float
#      - Carry **before** this observation, oldest to newest.
#    * - ``state.valid``
#      - ``[B, M]``, bool
#      - Usable historical positions, shared by the layers.
#    * - ``is_init``
#      - ``[B, 1]``, bool
#      - True on a real episode start; clear only those streams.
#    * - ``features`` (output)
#      - ``[B, D]``, float
#      - Representation for the action/value heads.
#    * - ``next.state`` (output)
#      - State with the same shapes
#      - Carry after this observation; becomes the next step's root state.
#
# ``L``, ``M`` and ``D`` are feature dimensions, never environment batch
# dimensions. Additional environment batch dimensions can precede them.
# Pass arbitrary nested keys through ``in_keys``/``out_keys``; the state output
# must be ``("next", *state_input_key)``. ``is_init`` remains the root key.


class GTrXLState(TypedTensorDict):
    """An example schema: shape, dtype and device belong to its leaf specs."""

    memory: torch.Tensor
    valid: torch.Tensor


torch.manual_seed(0)
torch.set_num_threads(1)
backbone = GTrXL(4, 32, num_layers=2, memory_len=8, state_cls=GTrXLState)
transformer = TransformerModule(
    transformer=backbone,
    in_keys=["observation", "state"],
    out_keys=["features", ("next", "state")],
)

######################################################################
# The schema is configurable: a plain ``TensorDict`` or a ``TensorClass`` with
# the same fields also works. This example does not establish a new default
# for recurrent modules. ``backbone.state_spec`` is an ordinary
# ``Composite(data_cls=GTrXLState)``: floating memory has explicit ``[L, M, D]``
# shape/device/dtype and validity has a boolean ``[M]`` leaf spec. No registry or
# new spec subclass is involved. Validate schemas during setup.

state = backbone.state_spec.zero([2])
backbone.state_spec.assert_is_in(state)
assert isinstance(state, GTrXLState)
assert state.batch_size == torch.Size([2])
assert state.memory.shape == (2, 2, 8, 32)

######################################################################
# Register state with the environment
# -----------------------------------
#
# ``InitTracker`` supplies ``is_init``. The primer adds state to the environment
# specs, initializes zero memory and clears only reset environments. This
# registration is necessary for state to survive stepping and collector buffers.
# Alternatively, ``Collector(..., auto_register_policy_transforms=True)`` installs
# the missing transforms. Here we register them explicitly for clarity.

actor_head = ProbabilisticActor(
    TensorDictModule(nn.Linear(32, 2), ["features"], ["logits"]),
    in_keys=["logits"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
)
critic_head = ValueOperator(nn.Linear(32, 1), in_keys=["features"])
policy = TensorDictSequential(transformer, actor_head, critic_head)
env = TransformedEnv(
    SerialEnv(2, ft.partial(GymEnv, "CartPole-v1", categorical_action_encoding=True)),
    InitTracker(),
)
env.set_seed(0)
env.append_transform(transformer.make_tensordict_primer())
collector = Collector(
    env,
    policy,
    frames_per_batch=32,
    total_frames=64,
    auto_register_policy_transforms=True,
)
rollouts = iter(collector)
next(rollouts)
# Use the second window, which can begin in the middle of an episode.
rollout = next(rollouts)
collector.shutdown()
assert rollout.batch_size == torch.Size([2, 16])
assert isinstance(rollout["state"], GTrXLState)

######################################################################
# Dense replay with separate memory and observation horizons
# ----------------------------------------------------------
#
# Root carry precedes time zero. In a compact record, time is a batch dimension
# of the **transitions child**, but not of the **outer record or initial state**:
#
# .. code-block:: text
#
#     record batch: [B]                 B independent windows
#       initial_state batch: [B]
#         memory: [B, L, M, D]          positions -M, ..., -1
#         valid:  [B, M]
#       transitions batch: [B, T]
#         observation: [B, T, F]        positions  0, ..., T-1
#         action, reward, done, ...
#
# The horizons meet at zero; increasing ``T`` does not duplicate the ``M`` slots.
# Storage capacity counts **windows**, so ``S = B*T`` transitions need
# ``B*L*M*D`` memory elements instead of ``S*L*M*D``. This removes a factor of
# ``T`` from the carry payload (and omits the extra ``next.state`` copy).
# Observations, actions and targets still require their usual per-step storage.
#
# This saving requires sampling whole records. A random window id is fine;
# an arbitrary new time offset inside that record needs a carry that is no
# longer stored. Do not flatten these records or apply ``SliceSampler`` to them.
# To support arbitrary slice starts, retain per-step states and use the ordinary
# ``[B, T]`` recurrent path. It loads the stored carry at the first position,
# even if ``is_init=False``, and at sampler-inserted ``is_init=True`` boundaries.
# With compact windows, ``is_init`` denotes **real episode resets only**.
#
# Compute advantages over the original rollout **before** sampling windows.
# The following short example evaluates all next values; the SOTA script reuses
# interior collected values and evaluates only episode/rollout boundaries.

with torch.no_grad():
    bootstrap = rollout["next"].clone()
    bootstrap["is_init"] = torch.zeros(2, 16, 1, dtype=torch.bool)
    TensorDictSequential(transformer, critic_head)(bootstrap)
    rollout["next", "state_value"] = bootstrap["state_value"]
    GAE(gamma=0.99, lmbda=0.95, value_network=None)(rollout)
    advantage = rollout["advantage"]
    rollout["advantage"] = (advantage - advantage.mean()) / advantage.std().clamp_min(
        1e-6
    )

transitions = rollout.select(
    "observation",
    "is_init",
    "action",
    "action_log_prob",
    "advantage",
    "value_target",
    "state_value",
    ("next", "reward"),
    ("next", "done"),
    ("next", "terminated"),
)
transitions["collector", "mask"] = torch.ones(2, 16, dtype=torch.bool)
records = TensorDict(
    {"initial_state": rollout["state"][:, 0].clone(), "transitions": transitions}, [2]
)
replay = TensorDictReplayBuffer(storage=LazyTensorStorage(2), batch_size=2)
replay.extend(records)
sample = replay.sample()

######################################################################
# Recompute features from the starting carry
# ------------------------------------------
#
# ``TransformerModule`` also accepts an outer record with batch ``[B]`` and
# observations ``[B, T, F]`` in recurrent mode. The single state remains ``[B]``.
# It returns features ``[B, T, D]`` and one final ``next.state`` with batch ``[B]``.
# No ``[B, T, L, M, D]`` state tensor is constructed on this path. GTrXL projects
# the initial memory once per layer and runs parallel causal attention over the
# window, masking episode crossings and positions more than ``M`` steps away.
# Output features can then be placed in the ``[B, T]`` transitions TensorDict
# that ordinary PPO heads and losses consume.

batch = sample["transitions"]
inputs = TensorDict(
    {
        "observation": batch["observation"],
        "is_init": batch["is_init"],
        "state": sample["initial_state"],
    },
    sample.batch_size,
)
with set_recurrent_mode(True):
    transformer(inputs)
batch["features"] = inputs["features"]
assert inputs["next", "state"].batch_size == sample.batch_size

loss_module = ClipPPOLoss(
    actor_head,
    critic_head,
    functional=False,
    normalize_advantage=False,
)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
losses = loss_module(batch)
loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
optimizer.step()

######################################################################
# Padding, staleness and memory cost
# ----------------------------------
#
# PPO automatically reads ``("collector", "mask")``. If a final window is
# shorter than ``T``, pad its transitions and set the mask false on padding;
# do not include padding in advantage normalization. Padding at the end cannot
# affect earlier causal outputs. The padded window's final carry includes its
# padded positions and must not be used to resume an actor. The SOTA script
# implements this packing and samples complete records without replacement.
#
# Initial memory is detached on entry. Gradients flow through recomputed
# activations within the window, including dependencies across time; they do
# not flow into the actor's saved carry. After optimizer updates, that carry
# may reflect older weights. GTrXL preserves it, just like stored GRU/LSTM
# state. A weight-update notification does not silently throw it away.
#
# This example compacts **after collection**. The replay allocation and compact
# learner input save memory; the standard collector still stores every step's
# carry before packing. Attention/activation memory also remains and grows with
# the window length. Do not interpret replay savings as collection-peak savings.
# Dropout is zero here: comparing collected and recomputed features otherwise
# includes differences from random dropout masks.
#
# Conclusion
# ----------
#
# A transformer policy needs observation features, episode markers and an action
# head. GTrXL additionally receives explicit pre-observation layer memory and
# writes its successor under ``next``. Whole-window replay preserves the initial
# carry once and recomputes the window, making the sampling/storage trade-off
# explicit. Existing cache-based transformers and GRU/LSTM defaults are unchanged.
#
# Further reading
# ---------------
#
# * :class:`~torchrl.modules.TransformerModule` and :class:`~torchrl.modules.GTrXL`.
# * :ref:`RNN_tuto` for per-step recurrent state and arbitrary slice sampling.
# * ``sota-implementations/gtrxl/README.md`` for hidden-velocity CartPole PPO,
#   Hydra configuration, padded windows and reproducible learning commands.
# * `Parisotto et al., Stabilizing Transformers for Reinforcement Learning
#   <https://proceedings.mlr.press/v119/parisotto20a.html>`_.
