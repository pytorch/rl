.. currentmodule:: torchrl

IsaacLab Integration
====================

.. _ref_isaaclab:

This guide covers how to use TorchRL components with
`IsaacLab <https://isaac-sim.github.io/IsaacLab/v2.3.0/>`_
(NVIDIA's GPU-accelerated robotics simulation platform).

For general IsaacLab installation and cluster setup (not specific to TorchRL), see the
`knowledge_base/ISAACLAB.md <https://github.com/pytorch/rl/blob/main/knowledge_base/ISAACLAB.md>`_ file.

IsaacLabWrapper
---------------

Use :class:`~torchrl.envs.libs.isaac_lab.IsaacLabWrapper` to wrap a gymnasium
IsaacLab environment into a TorchRL-compatible :class:`~torchrl.envs.EnvBase`:

.. code-block:: python

    import gymnasium as gym
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    env = gym.make("Isaac-Ant-v0", cfg=env_cfg)
    env = IsaacLabWrapper(env)

Key defaults:

- ``device=cuda:0``
- ``allow_done_after_reset=True``  (IsaacLab can report done immediately after reset)
- ``convert_actions_to_numpy=False``  (actions stay as tensors)

.. note::

    IsaacLab modifies ``terminated`` and ``truncated`` tensors in-place.
    ``IsaacLabWrapper`` clones these tensors to prevent data corruption.

.. note::

    Batched specs: IsaacLab env specs include the batch dimension (e.g., shape
    ``(4096, obs_dim)``).  Use ``*_spec_unbatched`` properties when you need
    per-env shapes.

.. note::

    Reward shape: IsaacLab rewards are ``(num_envs,)``.  The wrapper
    unsqueezes to ``(num_envs, 1)`` for TorchRL compatibility.

Collector
---------

Because IsaacLab environments are **pre-vectorized** (a single ``gym.make``
creates ~4096 parallel environments on the GPU), use a single
:class:`~torchrl.collectors.Collector` — there is no need for
``ParallelEnv`` or ``MultiCollector``:

.. code-block:: python

    from torchrl.collectors import Collector

    collector = Collector(
        create_env_fn=env,
        policy=policy,
        frames_per_batch=40960,   # 10 env steps * 4096 envs
        storing_device="cpu",
        no_cuda_sync=True,        # IMPORTANT for CUDA envs
    )

- ``no_cuda_sync=True``: avoids unnecessary CUDA synchronisation that can
  cause hangs with GPU-native environments.
- ``storing_device="cpu"``: moves collected data to CPU for the replay buffer.

2-GPU Async Pipeline
~~~~~~~~~~~~~~~~~~~~

For maximum throughput, use two GPUs with a background collection thread:

- **GPU 0 (``sim_device``)**: IsaacLab simulation + collection policy
  inference
- **GPU 1 (``train_device``)**: Model training (world model, actor, value
  gradients)

.. code-block:: python

    import copy, threading
    from tensordict import TensorDict

    # Deep copy policy to sim_device for collection
    collector_policy = copy.deepcopy(policy).to(sim_device)

    # Background thread for continuous collection
    def collect_loop(collector, replay_buffer, stop_event):
        for data in collector:
            replay_buffer.extend(data)
            if stop_event.is_set():
                break

    # Main thread: train on train_device
    for optim_step in range(total_steps):
        batch = replay_buffer.sample()
        train(batch)  # all on cuda:1
        # Periodic weight sync: training policy -> collector policy
        if optim_step % sync_every == 0:
            weights = TensorDict.from_module(policy)
            collector.update_policy_weights_(weights)

Key points:

- Both CUDA operations release the GIL, so they truly overlap.
- Must pass ``TensorDict.from_module(policy)`` to
  ``update_policy_weights_()``, not the module itself.
- Set ``CUDA_VISIBLE_DEVICES=0,1`` to expose 2 GPUs (IsaacLab defaults to
  only GPU 0).
- Falls back gracefully to single-GPU if only 1 GPU is available.

RayCollector (alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need distributed collection across multiple GPUs/nodes, use
:class:`~torchrl.collectors.distributed.RayCollector`:

.. code-block:: python

    from torchrl.collectors.distributed import RayCollector

    collector = RayCollector(
        [make_env] * num_collectors,
        policy,
        frames_per_batch=8192,
        collector_kwargs={
            "trust_policy": True,
            "no_cuda_sync": True,
        },
    )

Replay Buffer
-------------

The :class:`~torchrl.data.SliceSampler` needs enough sequential data.  With
``batch_length=50``, you need at least 50 time steps per trajectory before
sampling::

    init_random_frames >= batch_length * num_envs
                        = 50 * 4096
                        = 204,800

For GPU-resident replay buffers, use
:class:`~torchrl.data.LazyTensorStorage` with the target CUDA device.
This avoids CPU→GPU transfer at sample time (but adds it at extend time).

Dreamer-Specific Notes
----------------------

Observation key
~~~~~~~~~~~~~~~

IsaacLab uses ``"policy"`` as the observation key.  In Dreamer's pipeline,
this means:

- Encoder reads from ``"policy"`` (not ``"observation"`` or ``"pixels"``).
- Decoder writes to ``"reco_policy"``.
- Loss keys:
  ``world_model_loss.set_keys(pixels="policy", reco_pixels="reco_policy")``.

This is handled automatically when ``cfg.env.backend == "isaaclab"`` in
``make_dreamer()``.

No separate eval environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IsaacLab typically runs one simulation per process.  Creating a second env for
evaluation is unreliable.  Instead:

- Track episode rewards from completed episodes in the training data.
- The 4096 parallel envs provide statistically robust reward estimates.
- Periodic deterministic evaluation can be done by switching the policy's
  exploration type.

Throughput benchmarks
~~~~~~~~~~~~~~~~~~~~~

**Single GPU** (synchronous collect-then-train, NVIDIA H200, 4096 envs):

- ~15,600 fps data collection
- ~7.0 optim steps/sec (FP32, no compile)
- ~55 s for 200 steps end-to-end (3.6 ops/s)
- 50 % time collecting, 50 % training

**2-GPU async pipeline** (GPU 0 = sim, GPU 1 = train, bfloat16, torch.compile):

- ~78k–137k fps continuous collection
- ~2.9–3.5 optim steps/sec with 50k batch
- Collection and training fully overlap on separate GPUs
- ~9.5 h ETA for 100k steps on H200 (Anymal-C locomotion)

**Key insight**: with async 2-GPU, the replay buffer ``extend()`` (collector
thread) and ``sample()`` (training thread) both touch CPU memory, causing GIL
contention.  A GPU-resident replay buffer eliminates CPU-GPU transfer at sample
time.

Config differences from DMControl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 30

   * - Parameter
     - DMControl
     - IsaacLab
   * - ``env.backend``
     - ``dm_control``
     - ``isaaclab``
   * - ``env.from_pixels``
     - ``True``
     - ``False``
   * - ``collector.num_collectors``
     - ``7``
     - N/A (single Collector)
   * - ``collector.frames_per_batch``
     - ``1000``
     - ``204800``
   * - ``collector.init_random_frames``
     - ``10000``
     - ``204800``
   * - ``optimization.compile.enabled``
     - ``True``
     - ``True`` (sub-modules only)
   * - ``optimization.autocast``
     - ``bfloat16``
     - ``bfloat16``
   * - ``replay_buffer.batch_size``
     - ``10000``
     - ``50000``
   * - ``replay_buffer.gpu_storage``
     - N/A
     - ``true``

TorchRL-Specific Gotchas
------------------------

1. **``no_cuda_sync=True``**: Always set this for collectors with CUDA
   environments.  Without it, you get mysterious hangs.

2. **Installing torchrl in Isaac container**: Use
   ``--no-build-isolation --no-deps`` to avoid conflicts with Isaac's
   pre-installed torch/numpy.

3. **``TensorDictPrimer`` ``expand_specs``**: When adding primers (e.g.,
   ``state``, ``belief``) to a pre-vectorized env, you MUST pass
   ``expand_specs=True`` to :class:`~torchrl.envs.TensorDictPrimer`.
   Otherwise the primer shapes ``()`` conflict with the env's ``batch_size``
   ``(4096,)``.

4. **Model-based env spec double-batching**:
   ``model_based_env.set_specs_from_env(batched_env)`` copies specs with batch
   dims baked in.  The model-based env then double-batches actions during
   sampling (e.g., ``(4096, 4096, 8)`` instead of ``(4096, 8)``).

   **Fix**: unbatch the model-based env's specs after copying:

   .. code-block:: python

       model_based_env.set_specs_from_env(test_env)
       if test_env.batch_size:
           idx = (0,) * len(test_env.batch_size)
           model_based_env.__dict__["_output_spec"] = (
               model_based_env.__dict__["_output_spec"][idx]
           )
           model_based_env.__dict__["_input_spec"] = (
               model_based_env.__dict__["_input_spec"][idx]
           )
           model_based_env.empty_cache()

5. **``torch.compile`` with TensorDict**: Compiling full loss modules crashes
   because dynamo traces through TensorDict internals.  **Fix**: compile
   individual MLP sub-modules (encoder, decoder, reward_model, value_model)
   with ``torch._dynamo.config.suppress_errors = True``.  Do NOT compile RSSM
   (sequential, shared with collector) or loss modules (heavy TensorDict use).

6. **``SliceSampler`` with ``strict_length=False``**: The sampler may return
   fewer elements than ``batch_size``.  This causes
   ``reshape(-1, batch_length)`` to fail.

   **Fix**: truncate the sample:

   .. code-block:: python

       sample = replay_buffer.sample()
       numel = sample.numel()
       usable = (numel // batch_length) * batch_length
       if usable < numel:
           sample = sample[:usable]
       sample = sample.reshape(-1, batch_length)

7. **``frames_per_batch`` vs ``batch_length``**: Each collection adds
   ``frames_per_batch / num_envs`` time steps per env.  The
   ``SliceSampler`` needs contiguous sequences of at least ``batch_length``
   steps within a single trajectory.  Ensure
   ``frames_per_batch >= batch_length * num_envs`` for the initial collection,
   or that ``init_random_frames >= batch_length * num_envs``.

8. **``TD_GET_DEFAULTS_TO_NONE``**: Set this environment variable to ``1``
   when running inside the Isaac container to ensure correct TensorDict
   default behavior.
