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
        vla_action: VLAAction(...),                           # structured policy output
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
    VLAAction
    VLAImages
    VLAObservation
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
    VocabTailActionTokenizer

Image preprocessing
-------------------

The reusable :class:`~torchrl.data.vla.OpenVLAImagePreprocessor` implements the
OpenVLA-style image preprocessing order used by OpenVLA-OFT policies: square
resize, JPEG quality-95 round trip, optional 0.9-area center crop and resize
back. Its default backend keeps images as tensors and uses ``torchvision`` JPEG
codecs, while ``backend="pil"`` remains available as a reference path.

.. currentmodule:: torchrl.data.vla

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    OpenVLAImagePreprocessor

Policy and environment contract
-------------------------------

TorchRL keeps the environment boundary intentionally simple: base environments
do **not** need to know about a model-specific VLA wrapper or its structured
TensorClasses. The default env/policy contract is still the usual flat
TensorDict contract:

- on ``reset`` / ``step`` the environment writes observations under the
  canonical keys, usually ``("observation", "image")``, optional
  ``("observation", "state")`` and ``"language_instruction"``;
- on ``step`` the environment consumes one continuous action under
  ``"action"`` with shape ``[*B, action_dim]``;
- datasets and replay buffers may additionally carry training targets such as
  ``"action_chunk"``, ``"action_tokens"`` and ``"action_is_pad"``.

The VLA TensorClasses are structured policy/data containers, not a required env
API:

- :class:`~torchrl.data.vla.VLAImages` groups primary, wrist and extra camera
  tensors.
- :class:`~torchrl.data.vla.VLAObservation` lets a data pipeline store a
  structured or already-preprocessed observation under ``"vla_observation"``
  when a wrapper is built with ``input_mode="preprocessed"``.
- :class:`~torchrl.data.vla.VLAAction` mirrors policy outputs under
  ``"vla_action"`` for structured consumers, while the flat compatibility keys
  remain the default path for envs, transforms and losses.

In other words, a wrapper may read either canonical env keys
(``input_mode="canonical"``) or a preprocessed
:class:`~torchrl.data.vla.VLAObservation`
(``input_mode="preprocessed"``), but it always exposes flat outputs for
backward-compatible composition:

.. list-table:: Default policy outputs
    :header-rows: 1

    * - ``output_mode``
      - Flat keys
      - Structured mirror
    * - ``"chunk"``
      - ``"action_chunk"``
      - ``"vla_action".chunk``
    * - ``"tokens"``
      - ``"action_tokens"``, optional ``"log_probs"`` /
        ``"action_logits"`` / ``"action_mask"``
      - ``"vla_action".tokens``, ``.log_probs``, ``.logits``, ``.mask``
    * - ``"both"``
      - ``"action_chunk"`` and ``"action_tokens"`` plus optional token fields
      - Both ``"vla_action".chunk`` and ``"vla_action".tokens``

Chunked policies predict ``[*B, H, action_dim]`` while a standard env consumes
``[*B, action_dim]``. The bridge is explicit. Use
:class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper` when the env
should keep its one-step MDP: copy ``"action_chunk"`` to the env action key
``"action"``, let the wrapper cache the chunk, and re-query the expensive VLA
only when the cache expires or an ``"is_init"`` reset flag is observed. Use
:class:`~torchrl.envs.transforms.MultiAction` only when you want the env-side
transform to execute a whole chunk per policy call and accept the resulting
re-timed MDP.

Inference loop sketch
---------------------

The following pseudocode shows the default one-step env contract. The VLA
returns chunks, a one-line TensorDict module exposes them as the env action key,
and :class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper` serves one
cached action per environment step.

.. code-block:: python

    from tensordict.nn import TensorDictModule, TensorDictSequential
    from torchrl.envs import InitTracker, TransformedEnv
    from torchrl.modules import MultiStepActorWrapper
    from torchrl.modules.vla import TinyVLA

    H = 8
    base_env = make_robot_env()
    # base_env writes observation.image, observation.state and language_instruction
    env = TransformedEnv(base_env, InitTracker())  # writes is_init after reset

    policy = TinyVLA(
        action_dim=env.action_spec.shape[-1],
        chunk_size=H,
        output_mode="chunk",
    )

    chunk_as_env_action = TensorDictModule(
        lambda chunk: chunk,
        in_keys=["action_chunk"],
        out_keys=["action"],
    )
    actor = MultiStepActorWrapper(
        TensorDictSequential(policy, chunk_as_env_action),
        n_steps=H,
        replan_interval=None,  # open-loop: consume the full chunk before re-querying
    )

    td = env.reset()
    for _ in range(max_steps):
        td = actor(td)
        # td now contains:
        # - action_chunk: the predicted H-step chunk on re-plan steps
        # - vla_action: the structured mirror of the policy output
        # - action: the single env-facing action served from the cache
        td = env.step(td)
        td = td["next"]
        if td["done"].any():
            td = env.reset(td)

For token-output policies, request decoded chunks with ``output_mode="both"``
and pass an ``action_tokenizer`` to the wrapper. The env still consumes
``"action"``; token fields remain available for logging or RL fine-tuning.

Training loop sketch
--------------------

For offline chunked behavior cloning, the replay buffer stores canonical
observations and raw actions. Transforms build the training target
``"action_chunk"`` and its padding mask; the VLA wrapper reads observations and
predicts a chunk with the same flat key that :class:`~torchrl.objectives.BCLoss`
uses.

.. code-block:: python

    from torchrl.envs.transforms import ActionChunkTransform, Compose
    from torchrl.modules.vla import TinyVLA
    from torchrl.objectives import BCLoss

    H = 8
    replay_buffer = make_vla_replay_buffer(
        transform=Compose(
            # Optional: ActionScaling.from_metadata(...),
            ActionChunkTransform(chunk_size=H),
        )
    )

    policy = TinyVLA(action_dim=action_dim, chunk_size=H, output_mode="chunk")
    loss_module = BCLoss(policy, loss_function="l1")
    loss_module.set_keys(action="action_chunk", pad_mask="action_is_pad")

    optimizer = make_optimizer(policy.parameters())
    for _ in range(num_updates):
        batch = replay_buffer.sample(batch_size)
        loss_td = loss_module(batch)
        loss = loss_td["loss_bc"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Token RL fine-tuning uses the same policy contract. The rollout stores sampled
``"action_tokens"`` and behavior-policy ``"log_probs"``. During the update,
``get_dist`` / ``log_prob`` recompute the current policy probabilities from the
same observations, and :class:`~torchrl.objectives.ClipPPOLoss` consumes the
flat token keys.

.. code-block:: python

    from torchrl.modules.vla import TinyVLA
    from torchrl.objectives import ClipPPOLoss

    policy = TinyVLA(
        action_dim=action_dim,
        chunk_size=H,
        action_head="tokens",
        vocab_size=tokenizer.vocab_size,
        action_tokenizer=tokenizer,
        output_mode="both",  # tokens for the loss, decoded chunks for the env
        return_log_probs=True,
    )
    ppo_loss = ClipPPOLoss(policy, critic_network=None, entropy_bonus=False)
    ppo_loss.set_keys(
        action="action_tokens",
        sample_log_prob="log_probs",
        advantage="advantage",
    )

    for _ in range(num_updates):
        rollout = collector.next()  # contains sampled action_tokens and old log_probs
        rollout["advantage"] = compute_group_relative_advantage(rollout)
        # ClipPPOLoss calls the policy to recompute current token log-probs
        # against the stored actions, while log_probs remains the behavior
        # policy value collected during rollout.
        loss_td = ppo_loss(rollout)
        loss = loss_td["loss_objective"] + loss_td.get("loss_entropy", 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

If you write a custom token objective instead of using
:class:`~torchrl.objectives.ClipPPOLoss`, use :meth:`~torchrl.modules.vla.VLAWrapperBase.log_prob`
with a separate output key so the behavior-policy log-probabilities remain
available for ratios:

.. code-block:: python

    batch = policy.log_prob(batch, log_probs_key="new_log_probs")
    ratio = (batch["new_log_probs"] - batch["log_probs"]).exp()

Policies
--------

A VLA policy is an ordinary :class:`~tensordict.nn.TensorDictModuleBase` that
maps images, optional proprioceptive state and a language instruction to an
action chunk (continuous), action tokens (discrete), or both.
:class:`~torchrl.modules.vla.VLAWrapperBase` fixes that contract with explicit
``input_mode`` / ``output_mode`` settings, ``tensordict_out`` and
``logits_only`` forward paths, plus ``get_dist`` and ``log_prob`` methods for
loss-time recomputation. Existing flat keys (``action_chunk``,
``action_tokens``, ``action_logits``, ``action_mask`` and ``log_probs``) remain
the loss/transform compatibility path; the same values are mirrored into a
structured :class:`~torchrl.data.vla.VLAAction` under ``vla_action``.
:class:`~torchrl.modules.vla.TinyVLA` is a small reference policy for tests and
tutorials.

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
