.. currentmodule:: torchrl

.. _ref_recurrent_state_lifecycle:

Recurrent state lifecycle
=========================

Recurrent policies are not a special or dangerous path in TorchRL. In the
standard collection setup, most of the wiring is automated: passing a policy
to an environment, or constructing a collector with
``auto_register_policy_transforms=True``, lets TorchRL inspect the policy,
append :class:`~torchrl.envs.InitTracker`, and add the recurrent-state
primer required by :class:`~torchrl.modules.LSTMModule` or
:class:`~torchrl.modules.GRUModule`.

The main rule to keep in mind is simple: if the loss should replay
sequences, sample sequences. For replay-buffer training, use
:class:`~torchrl.data.SliceSampler` or another trajectory-aware sampler so
the loss receives contiguous time chunks with ``is_init`` boundaries
preserved. The rest of this page explains what the automated path wires up,
and what to check when building a custom loop, custom replay transform, or
manually constructed training batch.

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
            ├─ pad backend: split-and-pad when is_init[..., 1:] is set
            ├─ scan / triton backends: reset in place from is_init
            └─ write sequence outputs and trajectory-end hidden states back

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
- With the default eager ``"pad"`` backend, if any ``is_init`` in time
  positions ``1..T-1`` is true, the batch contains multiple trajectories
  packed together. The module calls ``_get_num_per_traj_init`` (see
  :func:`torchrl.objectives.value.utils._get_num_per_traj_init`) to count
  per-trajectory lengths, then ``_split_and_pad_sequence`` to break the
  batch into shape ``(N, T')`` with one trajectory per row.
- :class:`torch.nn.LSTM` is run on each clean row, then results are
  unpadded back to the original shape.
- With the ``"scan"`` or ``"triton"`` backends, the module keeps the
  original ``(B, T, ...)`` layout and uses ``is_init`` to reset hidden
  state in place instead of materializing split-and-padded chunks.
- All recurrent backends prevent hidden state from leaking *across*
  trajectories within a single training batch.

Mid-batch done
--------------

A "mid-batch done" is the case where, inside a single ``(B, T, ...)``
chunk, a trajectory ends at some ``t* < T-1`` and a new trajectory
starts at ``t*+1``. The corresponding ``is_init`` slot is true.

- In sequential collection this is handled step-by-step: at ``t*+1`` the
  policy sees ``is_init=True`` and zeros the hidden.
- In recurrent loss replay this is handled by the recurrent backend: the
  pad backend uses the split-and-pad path above, while the scan and triton
  backends reset hidden state directly from ``is_init``. **Without**
  ``is_init``, none of these paths sees the boundary and the LSTM treats
  the post-done timesteps as a continuation of the pre-done trajectory.

Hidden outputs and recurrent backends
-------------------------------------

TorchRL stores recurrent-state outputs with a time dimension, matching the
TensorDict batch shape. These outputs should be read as "next hidden"
checkpoints at trajectory ends, not as valid per-step hidden states at every
position.

With the pad backend, :class:`torch.nn.LSTM` only returns the final hidden
state for a sequence. :meth:`LSTMModule._lstm` therefore fills hidden slots
at non-terminal positions with zeros and writes the real final hidden at the
sequence end. If a batch contains multiple trajectories, the split path
packs or masks padded steps and writes the final hidden state back at each
real trajectory end.

With the scan and triton backends, this specific "``torch.nn.LSTM`` only
returns the final hidden" reason does not apply: those backends compute
hidden state along the original time dimension and reset it directly from
``is_init``. They still mask hidden outputs so the public TensorDict
semantics match the pad backend: trajectory-end slots contain usable hidden
checkpoints, while non-terminal slots are zero placeholders.

The advantage of the scan and triton backends is that mid-batch resets do
not require materializing split-and-padded trajectory chunks. This can reduce
VRAM use and improve throughput for batches with many resets. The
``"scan"`` backend is designed for :func:`torch.compile` friendliness, and
``recurrent_backend="auto"`` selects it when called under compilation. The
``"triton"`` backend uses CUDA kernels for the recurrent reset path. Some
configurations still fall back to the pad backend or are unsupported (for
example, scan does not support dropout, projections, or bidirectional LSTMs
yet), but the intended tradeoff is lower padding overhead plus better
compiled / fused execution.

Common debugging symptoms
-------------------------

**Symptom: reward looks fine but the policy never learns long-horizon behaviour.**
    Check that :class:`~torchrl.envs.InitTracker` is actually appended to
    the environment, and that ``is_init`` appears in the collected
    tensordict with true values at episode starts. A missing key usually
    raises quickly; a present-but-wrong all-false ``is_init`` signal is the
    silent failure mode.

**Symptom: training loss diverges or oscillates when you raise the batch's time horizon.**
    Likely hidden-state leakage across trajectory boundaries inside the
    replay batch. Use :class:`~torchrl.data.SliceSampler` or another
    sequence-aware sampler, verify that the recurrent loss path is wrapped
    in ``with set_recurrent_mode(True):``, and check that ``is_init`` is
    preserved through your replay buffer (some transforms drop unknown
    keys).

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

1. Use the automated path when possible: pass ``policy=`` to the env, or set
   ``auto_register_policy_transforms=True`` on the collector.
2. ``InitTracker`` is appended to the env, before any transform that might
   select keys.
3. ``is_init`` is present in the collected tensordict and is ``True`` on
   reset / immediately after a ``done``.
4. The recurrent state keys you pass to the LSTM module match the
   primer's keys (use :meth:`LSTMModule.make_tensordict_primer`).
5. Replay-buffer training uses :class:`~torchrl.data.SliceSampler` or
   another trajectory-aware sampler when the loss consumes sequences.
6. Loss / advantage code runs under ``with set_recurrent_mode(True):``.
7. The replay buffer preserves ``is_init`` (and any custom recurrent
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
