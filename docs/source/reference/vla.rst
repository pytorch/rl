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
TensorDict-native.

.. note::
    The VLA stack never hard-depends on the robot-learning ecosystem. Packages
    such as ``transformers``, ``lerobot`` or simulator backends are optional and
    imported lazily; ``import torchrl`` stays lightweight.

Canonical TensorDict schema
----------------------------

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

The default keys are exported from :mod:`torchrl.data.vla` (``IMAGE_KEY``,
``STATE_KEY``, ``INSTRUCTION_KEY``, ``ACTION_KEY``, ``ACTION_CHUNK_KEY``,
``ACTION_IS_PAD_KEY``, ``ACTION_TOKENS_KEY``). Every component also lets you
override its keys, so these are merely the shared defaults.

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

VLA-specific transforms are standard :class:`~torchrl.envs.transforms.Transform`
subclasses, so they compose with :class:`~torchrl.envs.transforms.Compose`,
replay buffers and transformed environments. They are documented in full on the
:ref:`transforms reference page <transforms>`.

- :class:`~torchrl.envs.transforms.ActionChunkTransform` -- build fixed-length
  action chunks (``[*B, T, H, action_dim]``) and a padding mask from a sampled
  trajectory window, the standard training target for chunked VLA policies.
- :class:`~torchrl.envs.transforms.ActionNormalize` -- affine action
  normalization (the action-space analogue of
  :class:`~torchrl.envs.transforms.ObservationNorm`); normalizes expert actions
  for training and denormalizes a policy's predicted actions for execution,
  with ``from_metadata`` / ``from_stats`` constructors.
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
one per step. :class:`~torchrl.modules.ActionChunkExecutor` -- a general
chunk-execution policy wrapper documented alongside the other actor modules --
wraps any chunk-predicting policy, emits one action per step and re-plans every
``replan_interval`` steps (receding horizon); it complements
:class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper`.

Objectives
----------

Fine-tuning objectives for VLA policies: chunked behavior cloning, and
reinforcement fine-tuning for token policies (GRPO / PPO-clip).

.. currentmodule:: torchrl.objectives.vla

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    VLABCLoss
    VLATokenGRPOLoss
