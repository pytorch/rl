"""
Vision-Language-Action (VLA) policies with TorchRL
==================================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _vla_tuto:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl

"""

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

##############################################################################
# What you will learn
# -------------------
#
# Vision-Language-Action (VLA) models map camera images, proprioceptive state
# and a natural-language instruction to robot actions -- usually emitted as a
# short *action chunk* of future steps. TorchRL treats a VLA as an ordinary
# TensorDict-first policy, so the same replay buffers, transforms, losses and
# collectors you already know apply.
#
# In this tutorial we will:
#
# - meet the canonical VLA TensorDict schema;
# - build action chunks and normalize actions with VLA transforms;
# - train a small reference policy by chunked behavior cloning;
# - execute the chunk policy one step at a time with a receding-horizon executor;
# - run one step of RL fine-tuning of a token policy with a GRPO objective.
#
# Everything below runs on CPU with synthetic data and a tiny model.

import torch
from tensordict import NonTensorStack, TensorDict

torch.manual_seed(0)

##############################################################################
# The canonical VLA schema
# ------------------------
#
# VLA components agree on a single key layout: camera image(s) and proprioceptive
# state live under ``observation``, while the per-trajectory language instruction
# and the action live at the tensordict root (mirroring
# :class:`~torchrl.data.datasets.OpenXExperienceReplay`). A single observation
# therefore looks like this:

batch, n_cam_c, hw, state_dim, action_dim = 8, 3, 16, 6, 4


def make_observation(batch=batch):
    return TensorDict(
        {
            "observation": {
                "image": torch.randint(
                    0, 255, (batch, n_cam_c, hw, hw), dtype=torch.uint8
                ),
                "state": torch.randn(batch, state_dim),
            },
            "language_instruction": NonTensorStack(
                *[f"pick up object {i}" for i in range(batch)]
            ),
        },
        batch_size=[batch],
    )


obs = make_observation()

##############################################################################
# Action chunking and normalization
# ----------------------------------
#
# :class:`~torchrl.envs.transforms.ActionChunkTransform` turns a per-step action
# tensor ``[*B, T, action_dim]`` into the chunked training target
# ``action_chunk`` ``[*B, T, H, action_dim]`` (plus an ``action_is_pad`` mask):
# for every step ``t`` it gathers the next ``H`` actions. This is the training
# target of modern chunked VLA policies (ACT, OpenVLA-OFT, pi0).

from torchrl.envs.transforms import ActionChunkTransform, ActionNormalize

T, H = 6, 4
window = TensorDict({"action": torch.randn(2, T, action_dim)}, batch_size=[2, T])
chunked = ActionChunkTransform(chunk_size=H)(window)
chunked["action_chunk"].shape  # [2, T, H, action_dim]

##############################################################################
# :class:`~torchrl.envs.transforms.ActionNormalize` is the action-space analogue
# of :class:`~torchrl.envs.transforms.ObservationNorm`: it normalizes expert
# actions for training and exposes :meth:`~torchrl.envs.transforms.ActionNormalize.denormalize`
# to map a policy's predicted action back to the raw action space for execution.

normalize = ActionNormalize(
    loc=torch.zeros(action_dim), scale=torch.ones(action_dim) * 2
)
normalized = normalize(TensorDict({"action": torch.full((4, action_dim), 2.0)}, [4]))
normalized["action"]  # all ones

##############################################################################
# A reference policy
# ------------------
#
# :class:`~torchrl.modules.vla.VLAWrapperBase` is the thin base class for VLA
# policies; :class:`~torchrl.modules.vla.TinyVLA` is a small reference policy
# (convolutional image encoder + state MLP + hashed instruction embedding +
# action head) for tutorials and tests. With a continuous head it predicts an
# action chunk.

from torchrl.modules.vla import TinyVLA

policy = TinyVLA(action_dim=action_dim, chunk_size=H, hidden_dim=64)
policy(make_observation())["action_chunk"].shape  # [batch, H, action_dim]

##############################################################################
# Behavior cloning
# ----------------
#
# :class:`~torchrl.objectives.vla.VLABCLoss` regresses the policy's predicted
# chunk onto an expert chunk (L1 by default), masking padded steps. Here we
# overfit a tiny synthetic dataset to confirm the policy learns.

from torchrl.objectives.vla import VLABCLoss

data = make_observation()
# a synthetic "expert": a fixed linear map from the state to an action chunk
expert = (
    data["observation", "state"] @ torch.randn(state_dim, H * action_dim)
).reshape(batch, H, action_dim)
data["action_chunk"] = expert

bc_loss = VLABCLoss(policy)
initial = bc_loss(data)[
    "loss_vla_bc"
].item()  # first call also materializes lazy params
optimizer = torch.optim.Adam(bc_loss.parameters(), lr=1e-2)
for _ in range(100):
    optimizer.zero_grad()
    bc_loss(data)["loss_vla_bc"].backward()
    optimizer.step()

##############################################################################
# The behavior-cloning loss drops sharply as the policy fits the expert chunks:

(initial, bc_loss(data)["loss_vla_bc"].item())

##############################################################################
# Chunked inference
# -----------------
#
# At inference, a chunk policy predicts ``H`` actions but the environment
# consumes one action per step. :class:`~torchrl.modules.ActionChunkExecutor`
# (a general chunk-execution policy wrapper, not VLA-specific) wraps the policy
# and emits one action per call, only re-invoking the (expensive) policy every
# ``replan_interval`` steps (receding horizon). Used as a collector or
# :meth:`~torchrl.envs.EnvBase.rollout` policy, it re-plans an environment when
# it is reset (``is_init``). Here we step it by hand to see the per-step actions.

from torchrl.modules import ActionChunkExecutor

executor = ActionChunkExecutor(policy, replan_interval=2)
actions = [executor(make_observation())["action"] for _ in range(5)]
# five [batch, action_dim] actions; the policy was only invoked on steps 0, 2, 4
[a.shape for a in actions]

##############################################################################
# RL fine-tuning
# --------------
#
# VLAs are increasingly post-trained with RL. A *token* VLA (action tokens
# emitted through a language-model head) can be fine-tuned with
# :class:`~torchrl.objectives.vla.VLATokenGRPOLoss`, a GRPO / PPO-clip objective
# over the action tokens with group-relative (or any precomputed) advantages and
# an optional KL penalty to a reference policy.
#
# We first roll out the token policy to obtain action tokens and their
# behavior-policy log-probabilities, attach a (here synthetic) advantage, then
# take one optimization step.

from torchrl.objectives.vla import VLATokenGRPOLoss

token_policy = TinyVLA(
    action_dim=action_dim,
    chunk_size=H,
    action_head="tokens",
    vocab_size=64,
    mode="sample",
)
rollout = token_policy(make_observation())  # writes action_tokens + log_probs
rollout["advantage"] = torch.randn(batch)
rollout["log_probs"] = rollout["log_probs"].detach()  # behavior log-probs are fixed

grpo_loss = VLATokenGRPOLoss(token_policy, clip_epsilon=0.2)
grpo_optimizer = torch.optim.Adam(grpo_loss.parameters(), lr=1e-3)
grpo_optimizer.zero_grad()
grpo_loss(rollout)["loss_objective"].backward()
grpo_optimizer.step()

##############################################################################
# Conclusion
# ----------
#
# We loaded VLA-shaped data into the canonical schema, built action chunks and
# normalized actions, trained a reference policy by chunked behavior cloning,
# executed it with a receding-horizon chunk executor, and ran one step of token
# GRPO fine-tuning -- all with the standard TorchRL primitives. To scale up,
# swap :class:`~torchrl.modules.vla.TinyVLA` for a wrapped open checkpoint
# (:class:`~torchrl.modules.vla.LeRobotPolicyWrapper`) and stream real data with
# :class:`~torchrl.data.datasets.LeRobotExperienceReplay` or
# :class:`~torchrl.data.datasets.OpenXExperienceReplay`.
#
# Further reading
# ---------------
#
# - OpenVLA-OFT (chunked continuous fine-tuning): https://arxiv.org/abs/2502.19645
# - pi0 (flow-matching VLA): https://arxiv.org/abs/2410.24164
# - FAST (action tokenization): https://arxiv.org/abs/2501.09747
# - SimpleVLA-RL (GRPO fine-tuning): https://arxiv.org/abs/2509.09674
# - The :ref:`VLA reference documentation <ref_vla>`.
