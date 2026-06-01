.. _ref_glossary:

Glossary
========

TorchRL borrows much of its vocabulary from :mod:`tensordict` and the broader
RL literature, but a handful of terms appear in error messages and source code
without a dedicated definition in the API reference. This page lists those
terms with the minimum context needed to find the relevant code.

.. glossary::

   _AcceptedKeys
      A dataclass nested inside most :class:`~torchrl.objectives.LossModule`
      subclasses that declares the tensordict keys the loss expects to read or
      write. Each field is a :class:`~tensordict.utils.NestedKey` with a
      default value. Override the defaults via
      :meth:`~torchrl.objectives.LossModule.set_keys` rather than mutating the
      dataclass directly; ``set_keys`` also propagates the change to the
      underlying value estimator.

   BatchedEnv
      A TorchRL environment that owns more than one environment instance under
      a single :class:`~torchrl.envs.EnvBase` interface. The common
      implementations are :class:`~torchrl.envs.SerialEnv` and
      :class:`~torchrl.envs.ParallelEnv`, both subclasses of
      :class:`~torchrl.envs.batched_envs.BatchedEnvBase`. Their ``batch_size`` is the
      leading shape of reset, step, and collector outputs.

   carrier
      The :class:`~tensordict.TensorDictBase` stored as ``self._carrier`` inside
      :meth:`~torchrl.collectors.SyncDataCollector.rollout`. It persists across
      collector batches and holds the post-reset environment output that the
      next policy call consumes. See :ref:`ref_collectors_internals` for the
      full lifecycle.

   Collector
      The single-process data collector, exposed as
      :class:`~torchrl.collectors.Collector`. It alternates policy calls and
      environment steps to produce rollout tensordicts; the legacy name
      :class:`~torchrl.collectors.SyncDataCollector` aliases the same
      implementation.

   compact_obs
      Collector setting that drops observation and state keys from the
      ``("next", ...)`` sub-tensordict of every persisted step. Within a
      contiguous same-trajectory sample, those values can be reconstructed from
      the root keys of the following step. At trajectory boundaries or in
      non-contiguous random samples, reconstruction must use the configured fill
      value; see :class:`~torchrl.envs.transforms.NextStateReconstructor` and the
      ``compact_obs`` argument on :class:`~torchrl.collectors.SyncDataCollector`.

   Composite
   CompositeSpec
      A nested spec container, currently named :class:`~torchrl.data.Composite`,
      that maps tensordict keys to leaf :class:`~torchrl.data.TensorSpec`
      objects. Environment specs such as ``observation_spec``, ``action_spec``,
      and ``reward_spec`` are usually composites. ``CompositeSpec`` is an older
      name that may still appear in discussions and issue reports.

   Env
      Short for environment: an object implementing the
      :class:`~torchrl.envs.EnvBase` API, including ``reset``, ``step``, specs,
      device handling, and a tensordict-based input/output contract. TorchRL env
      wrappers usually subclass :class:`~torchrl.envs.Transform` or compose a
      :class:`~torchrl.envs.TransformedEnv` rather than following the Gym
      wrapper API directly.

   env batch size
      The leading batch shape of an environment, exposed as
      :attr:`~torchrl.envs.EnvBase.batch_size`. A single unbatched env has an
      empty batch size; a :class:`~torchrl.envs.ParallelEnv` with ``N`` workers
      usually has batch size ``[N]``. Collectors append a time dimension to this
      shape when they stack rollout steps.

   env_device
      The collector device slot used for environment ``reset`` and ``step``
      operations. When it differs from ``policy_device`` or from the storage
      layout, the collector inserts the casts and sync points described in
      :ref:`ref_collectors_internals`.

   EnvCreator
      A small callable wrapper, :class:`~torchrl.envs.EnvCreator`, used to build
      environments lazily or in worker processes. It is useful when constructors
      need to be serialized for :class:`~torchrl.collectors.MultiSyncCollector`,
      :class:`~torchrl.collectors.MultiAsyncCollector`, or distributed
      collectors.

   functional (loss)
      A :class:`~torchrl.objectives.LossModule` is *functional* when it stores
      its actor / critic parameters as a stateless tensordict and invokes the
      networks with :meth:`~tensordict.TensorDictParams.to_module` at call time.
      This is what makes soft / target update, ``separate_losses=True``, and
      per-parameter optimiser groups possible without deep-copying the
      underlying ``nn.Module``. Check ``loss.functional`` to see which mode a
      given loss is in.

   in_keys
   out_keys
      The list of tensordict keys a module reads from (``in_keys``) and writes
      to (``out_keys``). Both :class:`~tensordict.nn.TensorDictModule` and most
      TorchRL loss / value-estimator components expose these as constructor
      arguments. Modifying them lets you wire a module into a tensordict layout
      that differs from the defaults; see :ref:`data_layout <ref_data_layout>`
      for naming conventions.

   is_init
      A boolean key (default name: ``"is_init"``) written by
      :class:`~torchrl.envs.InitTracker` immediately after every env reset.
      Recurrent modules and advantage estimators read this key to know where
      trajectories begin so they can zero out stale hidden state or reset the
      bootstrap target.

   no_cuda_sync
      A collector flag that suppresses the explicit CUDA, MPS, or NPU
      synchronizations inserted after cross-device transfers. Safe to set only
      when transfers are already correctly ordered or when running pure CPU.
      Defaults to ``False``.

   policy_device
      The collector device slot where the policy network runs. When it differs
      from ``env_device``, the collector casts the carrier before policy and env
      calls.

   recurrent mode
      The flag controlling whether an RNN-bearing module
      (:class:`~torchrl.modules.LSTMModule`,
      :class:`~torchrl.modules.GRUModule`) processes a single timestep per call
      (*sequential*) or a full ``(B, T, ...)`` sequence in one call
      (*recurrent*). Toggled via the
      :class:`~torchrl.modules.set_recurrent_mode` context manager. Collectors
      run in sequential mode; losses run in recurrent mode so the module can
      split and pad on trajectory boundaries inside a replayed batch.

   set_keys
      The public method on :class:`~torchrl.objectives.LossModule` and value
      estimators used to override the default tensordict keys a loss expects.
      Example: ``loss.set_keys(value=("agents", "state_value"),
      action=("agents", "action"))``. Prefer this over reaching into
      ``loss.tensor_keys`` directly because it
      also wires changes into the loss's value estimator if one exists.

   Specs
      Tensor constraints that describe valid values, shapes, dtypes, and
      devices. TorchRL uses :class:`~torchrl.data.TensorSpec` leaves, such as
      :class:`~torchrl.data.Bounded` and :class:`~torchrl.data.Unbounded`, and
      :class:`~torchrl.data.Composite` containers to validate and generate env
      inputs and outputs.

   storing_device
      The collector device slot where a rollout batch is materialised before it
      is yielded or extended into a replay buffer. Direct ``replay_buffer.add``
      writes bypass this materialisation path.

   TED
      TorchRL Episode Data: the standard offline dataset layout described in
      :ref:`TED-format`. It stores a transition with root keys for the current
      step and a ``("next", ...)`` sub-tensordict for next-step values.
      Conversion helpers such as :class:`~torchrl.data.TED2Flat` and
      :class:`~torchrl.data.Flat2TED` serialize and restore this layout.

   tensor_keys
      The instance attribute on every :class:`~torchrl.objectives.LossModule`
      holding the current values of the keys declared in ``_AcceptedKeys``.
      Read-only by convention; use
      :meth:`~torchrl.objectives.LossModule.set_keys` to modify them.

   TensorDictPrimer
      A :class:`~torchrl.envs.Transform` that injects keys into the
      environment's reset / step output that the policy needs but the env does
      not natively produce, most commonly RNN hidden states. Without a primer,
      the first call to a recurrent policy after reset would have no hidden
      state to read. See :class:`~torchrl.envs.TensorDictPrimer` and
      :meth:`torchrl.modules.LSTMModule.make_tensordict_primer`.

   trajectory ID
      An integer that uniquely identifies which trajectory each frame belongs
      to. Written by :class:`~torchrl.collectors.SyncDataCollector` as
      ``("collector", "traj_ids")`` when ``track_traj_ids=True``. Used by
      :class:`~torchrl.data.SliceSampler` to draw whole trajectories from a
      buffer and by :func:`~torchrl.collectors.utils.split_trajectories` to
      slice a flat batch into per-trajectory chunks.

   Transform
      TorchRL's tensordict-native environment transform abstraction,
      :class:`~torchrl.envs.Transform`. A transform can modify input specs,
      output specs, reset data, step data, or inverse action data, and is
      usually installed through :class:`~torchrl.envs.TransformedEnv`. This is
      distinct from a Gym wrapper, which operates on non-tensordict values.

See also
--------

- :ref:`ref_data_layout` — naming conventions for keys in collected batches
- :ref:`ref_collectors_internals` — where carrier / sync / device flags appear
  in the rollout loop
- :doc:`knowledge_base` — longer-form debugging notes
