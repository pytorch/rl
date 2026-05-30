.. currentmodule:: torchrl

.. _ref_recurrent_state_lifecycle:

Recurrent state lifecycle
=========================

Debugging a recurrent policy in TorchRL means understanding how the hidden
state flows from environment reset, through the policy and collector,
into the replay buffer, and finally back into the loss / advantage
computation. The pieces that make this work are spread across
:class:`~torchrl.envs.InitTracker`, :class:`~torchrl.modules.LSTMModule`
(and :class:`~torchrl.modules.GRUModule`),
:class:`~torchrl.modules.set_recurrent_mode`, and the ``is_init`` masking
inside loss code. This page traces the full path in one place.

The signal that ties it together is the ``"is_init"`` key: a boolean per
batch element that says "this is the first step of a fresh trajectory,
do not use the hidden state coming in." Every reset of recurrent state
in TorchRL ultimately ties back to this flag.

The path at a glance
--------------------

::

    env.reset() / done at step t
            │
            ▼
    InitTracker  ──────▶  sets is_init=True for that batch element
            │
            ▼
    rollout step (sequential mode)
        policy(tensordict)
            │
            ▼
        LSTMModule.forward (sequential)
            │
            ├─ reads hidden from tensordict   (zeros at reset, prev step otherwise)
            ├─ where is_init=True, zeros it   ◀── boundary reset happens here
            └─ writes next-step hidden into ("next", "rs_h"), ("next", "rs_c")
            │
            ▼
    SyncDataCollector
            │
            ├─ step_mdp moves ("next", "rs_*") to the root for step t+1
            └─ emits a batched TensorDict of shape (B, T, ...)
            │
            ▼
    Replay buffer    (stores (B, T, ...) trajectories with is_init preserved)
            │
            ▼
    Loss / GAE       (recurrent mode)
        with set_recurrent_mode(True):
            value_net(sampled_batch)
            │
            ▼
        LSTMModule.forward (recurrent)
            │
            ├─ if any is_init[..., 1:] set:
            │     split-and-pad along trajectory boundaries
            ├─ run nn.LSTM on each clean (B', T') chunk
            └─ unpad and write outputs back

What ``is_init`` means
----------------------

``is_init`` is a boolean key shaped like the env's batch (``(*B, 1)``),
set by :class:`~torchrl.envs.InitTracker` to ``True`` on the *first* step
of every trajectory and ``False`` everywhere else. A trajectory begins
at an explicit :meth:`~torchrl.envs.EnvBase.reset` or right after a
``done`` from the previous step.

If you do not append :class:`~torchrl.envs.InitTracker` to your env,
``is_init`` will be absent and :class:`~torchrl.modules.LSTMModule` will
raise a ``KeyError``. If the key is present but always ``False`` (or if a
custom replay buffer / transform drops or rewrites the true boundary
signal), the LSTM has no way to know when a new trajectory has started.
In that case the hidden state will silently carry forward across episode
boundaries — usually the most painful class of recurrent bug to diagnose
because rewards still look plausible.

When hidden state resets vs. is carried forward
-----------------------------------------------

There are two execution modes, gated by
:class:`~torchrl.modules.set_recurrent_mode` and the module's
``default_recurrent_mode``:

**Sequential mode** (``set_recurrent_mode(False)``, the default during
collection):

- The policy is called once per environment step.
- :meth:`~torchrl.modules.LSTMModule.forward` reads the incoming hidden
  from the root tensordict and, for any batch element where
  ``is_init=True``, replaces it with zeros before running the LSTM cell::

      is_init_expand = expand_as_right(is_init, hidden0)
      hidden0 = torch.where(is_init_expand, zeros, hidden0)
      hidden1 = torch.where(is_init_expand, zeros, hidden1)

- The new hidden is written under the ``("next", ...)`` keys and
  :meth:`~torchrl.envs.utils.step_mdp` promotes it to the root for the
  following step. This is how the carry-forward happens between
  non-boundary steps.

**Recurrent mode** (``set_recurrent_mode(True)``, the default inside
TorchRL loss / advantage code):

- A full ``(B, T, ...)`` batch is passed in one call.
- If any ``is_init`` in time positions ``1..T-1`` is true, the batch
  contains multiple trajectories packed together. The module calls
  ``_get_num_per_traj_init`` (see
  :func:`torchrl.objectives.value.utils._get_num_per_traj_init`) to count
  per-trajectory lengths, then ``_split_and_pad_sequence`` to break the
  batch into shape ``(N, T')`` with one trajectory per row.
- :class:`torch.nn.LSTM` is run on each clean row, then results are
  unpadded back to the original shape. This is what keeps hidden state
  from leaking *across* trajectories within a single training batch.

Mid-batch done
--------------

A "mid-batch done" is the case where, inside a single ``(B, T, ...)``
chunk, a trajectory ends at some ``t* < T-1`` and a new trajectory
starts at ``t*+1``. The corresponding ``is_init`` slot is true.

- In sequential collection this is handled step-by-step: at ``t*+1`` the
  policy sees ``is_init=True`` and zeros the hidden.
- In recurrent loss replay this is handled by the split-and-pad path
  above. **Without** ``is_init``, the split never fires and the LSTM
  treats the post-done timesteps as a continuation of the pre-done
  trajectory.

Why the final hidden values should not always be trusted
--------------------------------------------------------

:meth:`LSTMModule._lstm` pads the per-step hidden outputs because
:class:`torch.nn.LSTM` only returns the final hidden across the whole
sequence, while tensordict expects a hidden value at every step. The
intermediate steps are zero-padded; the *last* step contains the
real final hidden.

In recurrent mode with multi-trajectory splitting, the current split path
packs or masks padded steps and writes the final hidden state back at the
real trajectory end. Hidden entries at non-terminal steps are still
zero-padding placeholders rather than per-step recurrent states. The
practical consequence is that you should only consume hidden outputs from
trajectory-end positions (or carry state step-by-step during collection),
not treat every time step's hidden slot as a valid starting point for a
follow-on rollout.

Common debugging symptoms
-------------------------

**Symptom: reward looks fine but the policy never learns long-horizon behaviour.**
    Check that :class:`~torchrl.envs.InitTracker` is actually appended
    to the environment, and that ``is_init`` appears in the collected
    tensordict. A missing transform is silent.

**Symptom: training loss diverges or oscillates when you raise the batch's time horizon.**
    Likely hidden-state leakage across trajectory boundaries inside the
    replay batch. Verify that the recurrent loss path is wrapped in
    ``with set_recurrent_mode(True):`` and that ``is_init`` is preserved
    through your replay buffer (some transforms drop unknown keys).

**Symptom: shapes mismatch in** :meth:`LSTMModule._lstm` **with cryptic transpose errors.**
    The module expects the tensordict-native hidden layout
    ``(batch, steps, num_layers, hidden_size)``. A custom
    :class:`~torchrl.envs.transforms.TensorDictPrimer` with a different
    shape, or a manually-constructed hidden, will fail here. Prefer
    :meth:`LSTMModule.make_tensordict_primer` to avoid drift.

**Symptom: "fresh" trajectory inherits the previous episode's behaviour.**
    Either ``is_init`` is not being set at the right step (check
    :class:`InitTracker`'s placement relative to other transforms that
    might reset state), or you are reusing a final hidden as a starting
    state across rollouts (see the previous section).

**Symptom: identical results regardless of** ``set_recurrent_mode`` **value.**
    Check whether the call actually runs inside the context manager you
    expect, and whether another nested ``set_recurrent_mode`` context is
    overriding it. The module's ``default_recurrent_mode`` is only used
    when no context manager is active.

What to check, in order
-----------------------

1. ``InitTracker`` is appended to the env, before any transform that
   might select keys.
2. ``is_init`` is present in the collected tensordict and is ``True`` on
   reset / immediately after a ``done``.
3. The recurrent state keys you pass to the LSTM module match the
   primer's keys (use :meth:`LSTMModule.make_tensordict_primer`).
4. Loss / advantage code runs under ``with set_recurrent_mode(True):``.
5. The replay buffer preserves ``is_init`` (and any custom recurrent
   keys) through its transforms.

See also
--------

- :class:`~torchrl.modules.LSTMModule` — the module that consumes
  ``is_init`` and gates hidden-state resets.
- :class:`~torchrl.modules.GRUModule` — same lifecycle, single hidden
  state.
- :class:`~torchrl.modules.set_recurrent_mode` — context manager for
  switching execution paths.
- :class:`~torchrl.envs.InitTracker` — the source of ``is_init``.
- :func:`torchrl.objectives.value.utils._get_num_per_traj_init` and
  :func:`torchrl.objectives.value.functional._split_and_pad_sequence` —
  the trajectory-boundary plumbing.
