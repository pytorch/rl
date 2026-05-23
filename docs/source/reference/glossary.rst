.. _ref_glossary:

Glossary
========

TorchRL borrows much of its vocabulary from :mod:`tensordict` and the broader
RL literature, but a handful of terms appear in error messages and source
code without a dedicated definition in the API reference.  This page lists
those terms with the minimum context needed to find the relevant code.

.. glossary::
   :sorted:

   carrier
   shuttle
      The single :class:`~tensordict.TensorDictBase` instance that survives
      across timesteps inside :meth:`~torchrl.collectors.SyncDataCollector.rollout`
      and carries data between the environment and the policy.  Stored as
      ``self._carrier`` (the older name ``shuttle`` is kept in some
      comments).  See :ref:`ref_collectors_internals` for the full lifecycle.

   in_keys
   out_keys
      The list of tensordict keys a module *reads from* (``in_keys``) and
      *writes to* (``out_keys``).  Both :class:`~tensordict.nn.TensorDictModule`
      and most of TorchRL's loss / value-estimator components expose these
      as constructor arguments.  Modifying them lets you wire a module into
      a tensordict layout that differs from the defaults — see
      :class:`~tensordict.nn.TensorDictModule` and the
      :ref:`data_layout <ref_data_layout>` page for naming conventions.

   _AcceptedKeys
      A dataclass nested inside most :class:`~torchrl.objectives.LossModule`
      subclasses that declares the tensordict keys the loss expects to read
      or write.  Each field is a :class:`~tensordict.utils.NestedKey` with a
      default value.  Override the defaults via
      :meth:`~torchrl.objectives.LossModule.set_keys` rather than mutating
      the dataclass directly — ``set_keys`` also propagates the change to
      the underlying value estimator.

   set_keys
      The public method on :class:`~torchrl.objectives.LossModule` (and on
      value estimators) used to override the default tensordict keys a loss
      expects.  Example: ``loss.set_keys(value=("agents", "state_value"),
      action=("agents", "action"))``.  Prefer this over reaching into
      ``loss.tensor_keys`` directly because it also wires the changes into
      the loss's value estimator if one exists.

   recurrent mode
      The flag controlling whether an RNN-bearing module
      (:class:`~torchrl.modules.LSTMModule`,
      :class:`~torchrl.modules.GRUModule`) treats the time dimension
      sequentially or one step at a time.  Toggled per-call via
      :meth:`~torchrl.modules.LSTMModule.set_recurrent_mode` or globally via
      the ``recurrent_mode_state_manager`` context manager.  Collectors set
      it to step-by-step during rollout; losses set it to sequential during
      backprop over a trajectory chunk.

   TensorDictPrimer
      A :class:`~torchrl.envs.Transform` that injects keys into the
      environment's reset / step output that the policy needs but the env
      does not natively produce — most commonly RNN hidden states.  Without
      a primer, the first call to a recurrent policy after reset would have
      no hidden state to read.  See
      :class:`~torchrl.envs.TensorDictPrimer` and
      :meth:`torchrl.modules.LSTMModule.make_tensordict_primer`.

   is_init
      A boolean key (default name: ``"is_init"``) written by
      :class:`~torchrl.envs.InitTracker` immediately after every env reset.
      Recurrent modules and advantage estimators read this key to know
      where trajectories begin so they can zero out stale hidden state or
      reset the bootstrap target.  If ``is_init`` is missing or wrongly
      wired, hidden state from a previous trajectory will leak into the
      next — a class of bug that looks like a learning regression rather
      than a key-routing error.

   trajectory ID
      An integer that uniquely identifies which trajectory each frame
      belongs to.  Written by
      :class:`~torchrl.collectors.SyncDataCollector` as
      ``("collector", "traj_ids")`` when ``track_traj_ids=True``.  Used by
      :class:`~torchrl.data.SliceSampler` to draw whole trajectories from
      a buffer and by :func:`~torchrl.collectors.utils.split_trajectories`
      to slice a flat batch into per-trajectory chunks.

   storing_device
   policy_device
   env_device
      The three device slots a collector tracks separately.  ``policy_device``
      is where the policy network lives; ``env_device`` is where the
      environment's step / reset run; ``storing_device`` is where the
      collected batch is materialised before being yielded.  When any two
      differ, the collector inserts explicit ``.to(...)`` casts and the
      matching ``_sync_*`` call — see :ref:`ref_collectors_internals`.

   no_cuda_sync
      A collector flag that suppresses the explicit
      ``torch.cuda.synchronize`` (or MPS/NPU equivalent) inserted after
      cross-device transfers.  Safe to set only when all transfers are
      CUDA-stream-ordered or when running pure-CPU.  Defaults to ``False``.

   compact_obs
      Collector setting that drops observation keys from the ``("next",
      ...)`` sub-tensordict of every persisted step.  The observations are
      recoverable from the root keys of the *following* step, so the
      collected batch is smaller without information loss — at the cost
      of indirection at sampling time.  See the ``compact_next_keys``
      argument on :class:`~torchrl.collectors.SyncDataCollector`.

   functional (loss)
      A :class:`~torchrl.objectives.LossModule` is *functional* when it
      stores its actor / critic parameters as a stateless tensordict and
      invokes the networks with :meth:`~tensordict.TensorDictParams.to_module`
      at call time.  This is what makes ``soft / target update``,
      ``separate_losses=True``, and per-parameter optimiser groups possible
      without deep-copying the underlying ``nn.Module``.  Check ``loss.functional``
      to see which mode a given loss is in.

   tensor_keys
      The instance attribute on every :class:`~torchrl.objectives.LossModule`
      holding the *current* values of the keys declared in ``_AcceptedKeys``.
      Read-only by convention — use :meth:`~torchrl.objectives.LossModule.set_keys`
      to modify them.

See also
--------

- :ref:`ref_data_layout` — naming conventions for keys in collected batches
- :ref:`ref_collectors_internals` — where carrier / sync / device flags
  appear in the rollout loop
- :doc:`knowledge_base` — longer-form debugging notes
