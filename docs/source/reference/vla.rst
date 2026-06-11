.. currentmodule:: torchrl

Vision-Language-Action (VLA)
============================

.. _ref_vla:

Vision-Language-Action (VLA) models map one or more camera images,
optional proprioceptive state, and a natural-language instruction to robot
actions -- usually emitted as a short *action chunk* of future steps. TorchRL
treats a VLA as an ordinary TensorDict-first policy: a :class:`~tensordict.nn.TensorDictModule`
fed by composable transforms, trained by a :class:`~torchrl.objectives.LossModule`,
and rolled out by the standard collectors. This page documents the data
schema, transforms, policies and objectives that make robot VLA workflows
TensorDict-native. See the :ref:`VLA tutorial <vla_tuto>` for an end-to-end
example (data, chunking, behavior cloning, chunked inference and RL fine-tuning).

.. note::
    The VLA stack never hard-depends on the robot-learning ecosystem. Packages
    such as ``transformers``, ``lerobot`` or simulator backends are optional and
    imported lazily; ``import torchrl`` stays lightweight.

Canonical TensorDict schema
---------------------------

VLA components agree on a single :class:`~tensordict.utils.NestedKey` layout so
that datasets, transforms, policies and losses interoperate without lossy
conversion. The layout mirrors :class:`~torchrl.data.datasets.OpenXExperienceReplay`
and the LeRobot dataset format::

    TensorDict(
        observation: TensorDict(
            image: {<camera>: uint8/float [*B, T, C, H, W]},  # or a single tensor
            state: float [*B, T, state_dim],                  # proprioception
        ),
        language_instruction: NonTensorData | Text,           # raw or tokenized (per-traj)
        action: float [*B, T, action_dim],                    # raw, per-step
        action_chunk: float [*B, T, chunk, action_dim],       # built for training
        action_is_pad: bool [*B, T, chunk],                   # chunk validity mask
        action_tokens: long [*B, T, chunk, action_dim],       # tokenized actions
        next: TensorDict(...),                                 # TED layout
    )

Like :class:`~torchrl.data.datasets.OpenXExperienceReplay`, the image and state
live under ``observation`` while the (per-trajectory) language instruction and
the action live at the tensordict root.

The default keys are exported from :mod:`torchrl.data.vla` (``OBSERVATION_KEY``,
``IMAGE_KEY``, ``STATE_KEY``, ``INSTRUCTION_KEY``, ``ACTION_KEY``,
``ACTION_CHUNK_KEY``, ``ACTION_IS_PAD_KEY``, ``ACTION_TOKENS_KEY``). Every
component also lets you override its keys, so these are merely the shared
defaults.

Data and metadata
-----------------

.. currentmodule:: torchrl.data.vla

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    RobotDatasetMetadata
    validate_vla_tensordict

Robot VLA trajectories can be loaded into the canonical schema from
:class:`~torchrl.data.datasets.OpenXExperienceReplay` (Open X-Embodiment) and
:class:`~torchrl.data.datasets.LeRobotExperienceReplay` (the LeRobot format),
both of which expose trajectory-aware slice sampling.

Transforms
----------

The VLA data path is built from general :class:`~torchrl.envs.transforms.Transform`
subclasses -- none of them are VLA-specific (they apply to any action-based
pipeline) and they live alongside the other transforms, documented in full on the
:ref:`transforms reference page <transforms>`. Here is how they combine for VLA:

- :class:`~torchrl.envs.transforms.ActionChunkTransform` -- build fixed-length
  action chunks (``[*B, T, H, action_dim]``) and a padding mask from a sampled
  trajectory window, the standard training target for chunked VLA policies.
- :class:`~torchrl.envs.transforms.ActionScaling` -- affine action
  normalization; built with the
  :meth:`~torchrl.envs.transforms.ActionScaling.from_metadata` /
  :meth:`~torchrl.envs.transforms.ActionScaling.from_stats` constructors it
  normalizes expert actions on the replay-buffer sample path (pass
  ``in_keys_inv=[]`` for a buffer that raw data is written to through
  ``extend``, which applies the inverse) and denormalizes a policy's predicted
  actions on the env action-input path.
- :class:`~torchrl.envs.transforms.ActionTokenizerTransform` -- encode
  continuous actions into discrete tokens (wrapping an action tokenizer) for
  autoregressive token VLAs.
- :class:`~torchrl.envs.transforms.SuccessReward` -- a sparse 0/1 success
  reward for RL fine-tuning.

Action representations
----------------------

Action tokenizers map continuous actions to discrete token ids and back, so
that autoregressive (RT-2 / OpenVLA-style) VLA policies can emit actions through
a language-model head.

.. currentmodule:: torchrl.data.vla

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ActionTokenizerBase
    UniformActionTokenizer

Policies
--------

A VLA policy is an ordinary :class:`~tensordict.nn.TensorDictModuleBase` that
maps images, optional proprioceptive state and a language instruction to an
action chunk (continuous) or action tokens (discrete). :class:`~torchrl.modules.vla.VLAWrapperBase`
fixes that contract; :class:`~torchrl.modules.vla.TinyVLA` is a small reference
policy for tests and tutorials.

.. currentmodule:: torchrl.modules.vla

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    VLAWrapperBase
    TinyVLA
    LeRobotPolicyWrapper

At inference a chunk policy predicts ``H`` actions while the environment consumes
one per step -- and chunking only pays off if the (expensive) policy is *not*
queried at every step.
:class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper` provides
this: it caches the predicted actions, emits one per step and skips the
wrapped actor while the cache lasts -- open-loop by default, receding horizon
with ``replan_interval``, re-planning on env resets via ``is_init``.
:class:`~torchrl.envs.transforms.MultiAction` is the env-side alternative
(one base step per chunk action, a single policy call per chunk, at the price
of a re-timed MDP). :class:`~torchrl.envs.ToyVLAEnv` -- a tiny synthetic env
speaking the canonical schema, whose state echoes the executed action -- lets
you smoke-test this machinery without any simulator dependency.

Objectives
----------

VLA fine-tuning needs no dedicated loss classes; the standard objectives
apply directly:

- *Chunked behavior cloning* is :class:`~torchrl.objectives.BCLoss` with the
  action chunk as the ``action`` and the chunk-padding mask excluded via its
  ``pad_mask`` key::

      loss = BCLoss(policy, loss_function="l1")
      loss.set_keys(action="action_chunk", pad_mask="action_is_pad")

- *Token RL fine-tuning* (GRPO-style, following SimpleVLA-RL / RL4VLA) is
  :class:`~torchrl.objectives.ClipPPOLoss` over the action tokens: advantages
  are precomputed (group-relative), so no critic is needed, and the token
  head's sequence-level log-probabilities match the ``sample_log_prob``
  contract. The advantage carries the trailing singleton value-dim the PPO
  losses expect (``[batch, 1]``, not ``[batch]``)::

      loss = ClipPPOLoss(policy, critic_network=None, entropy_bonus=False)
      loss.set_keys(
          action="action_tokens", sample_log_prob="log_probs", advantage="advantage"
      )
