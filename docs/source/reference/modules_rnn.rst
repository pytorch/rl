.. currentmodule:: torchrl.modules

Recurrent modules
=================

TorchRL recurrent modules wrap PyTorch RNNs in TensorDict-aware modules.
:class:`LSTMModule` and :class:`GRUModule` read observations and recurrent
state entries from a :class:`~tensordict.TensorDict`, write features and
``("next", ...)`` recurrent states back to it, and use the ``is_init`` key to
reset hidden states at trajectory boundaries.

The recurrent modules are designed for the contiguous trajectory layout
described in :ref:`data-layout`: replay buffers and samplers can return flat
1-D slices, and the modules recover the sequence boundaries from ``is_init``
instead of requiring padded ``[B, T]`` tensors and masks.

Execution modes
---------------

By default, recurrent modules run in single-step mode. This is the mode used
during environment interaction: the input TensorDict contains one step per
environment, the previous recurrent state is read from the TensorDict, and the
next recurrent state is written under ``("next", ...)``.

During training, wrap the recurrent policy with
:func:`set_recurrent_mode` to process complete rollouts or replay-buffer
slices:

.. code-block:: python

    from torchrl.modules import GRUModule, set_recurrent_mode

    gru = GRUModule(
        input_size=4,
        hidden_size=64,
        in_keys=["observation", "recurrent_state", "is_init"],
        out_keys=["features", ("next", "recurrent_state")],
    )

    with set_recurrent_mode("recurrent"):
        batch = gru(batch)

In recurrent mode, every ``is_init=True`` entry resets the hidden state to the
state stored at that same position. This lets a flat batch of concatenated
trajectory slices behave like independent sequences without materializing
padding.

Backend selection
-----------------

The ``recurrent_backend`` constructor argument controls how recurrent-mode
calls handle resets inside a batch.

``"pad"``
    Splits trajectories on ``is_init``, pads them to a common length, and uses
    PyTorch's cuDNN-backed :class:`torch.nn.LSTM` or :class:`torch.nn.GRU`.
    This is the default and the broadest compatibility path.

``"scan"``
    Uses a scan over the time dimension and avoids padded trajectory chunks.
    This is friendlier to :func:`torch.compile` for reset-heavy RL batches.
    Supports unidirectional GRU/LSTM without dropout, and (for LSTM) without
    projections. Unsupported configurations raise when the recurrent path is
    executed.

``"triton"``
    Uses TorchRL's fused Triton kernels for reset-aware GRU/LSTM recurrence.
    This backend is CUDA-only and requires a recent Triton installation. It is
    intended for reset-heavy recurrent RL training where split/pad overhead is
    significant. Multilayer unidirectional modules (including dropout between
    layers) are handled directly; unsupported variants — bidirectional modules
    and LSTM projections — silently fall back to the pad semantics.

``"auto"``
    Uses ``"pad"`` in eager mode and ``"scan"`` when called under
    :func:`torch.compile`.

For long-running experiments, prefer choosing a backend explicitly once the
model shape and deployment target are known. ``"pad"`` is the safest baseline,
``"scan"`` is the compile-friendly baseline, and ``"triton"`` is the
performance-oriented CUDA backend.

Triton precision controls
-------------------------

The Triton backend performs hidden-to-hidden recurrent matrix multiplications
inside Triton kernels and input-to-hidden projections through PyTorch/cuBLAS.
The ``recurrent_matmul_precision`` argument keeps those paths aligned.

Supported values are:

``"auto"``
    Defer to the process-wide TorchRL setting, and fall back to
    :func:`torch.get_float32_matmul_precision` if the global is itself
    ``"auto"``. The ``TORCHRL_RNN_PRECISION`` environment variable seeds the
    process-wide setting at import time. It is not consulted at every kernel
    call; call :func:`set_recurrent_matmul_precision` with ``"auto"`` or
    ``None`` to re-read it after import.

``"ieee"``
    Use IEEE FP32 matmuls (~23 bits of mantissa, CUDA cores, no tensor
    cores). This is the most conservative setting and is useful for numerical
    comparisons with the scan backend.

``"tf32"``
    Use TF32 tensor cores on Ampere or newer NVIDIA GPUs (~10 bits of
    mantissa, highest throughput).

``"tf32x3"``
    Use Triton's three-product TF32 decomposition for the recurrent matmul
    (~22 bits of mantissa on tensor cores). cuBLAS has no ``tf32x3`` mode, so
    the input-to-hidden projection stays IEEE FP32. Useful when long rollouts
    make recurrent precision drift visible.

``"fast"`` and ``"high-prec"``
    GPU-aware presets. On TF32-capable NVIDIA GPUs, ``"fast"`` resolves to
    ``"tf32"`` and ``"high-prec"`` resolves to ``"tf32x3"``. On devices
    without TF32 tensor cores, both resolve to ``"ieee"``.

The process-wide default can be changed with
:func:`set_recurrent_matmul_precision`:

.. code-block:: python

    from torchrl.modules import set_recurrent_matmul_precision

    set_recurrent_matmul_precision("high-prec")
    gru = GRUModule(
        input_size=4,
        hidden_size=64,
        recurrent_backend="triton",
        recurrent_matmul_precision="auto",
        in_keys=["observation", "recurrent_state", "is_init"],
        out_keys=["features", ("next", "recurrent_state")],
    )

A module-level ``recurrent_matmul_precision=...`` value takes precedence over
the process-wide setting. Use :func:`get_recurrent_matmul_precision` to inspect
the resolved concrete mode for the current device.

Choosing a layout and backend
-----------------------------

For most recurrent RL pipelines:

* Use :class:`~torchrl.envs.transforms.InitTracker` or pass the policy to the
  env/collector so that TorchRL adds the ``is_init`` key and recurrent-state
  primers automatically.
* Store replay data in the flat contiguous layout and sample with
  :class:`~torchrl.data.replay_buffers.SliceSampler`.
* Run collection in single-step mode and training under
  :func:`set_recurrent_mode`.
* Start with ``recurrent_backend="pad"`` for correctness, then benchmark
  ``"scan"`` or ``"triton"`` for the target hardware.

See also
--------

* :ref:`data-layout` for the contiguous trajectory layout and replay-buffer
  handoff.
* :class:`LSTMModule` and :class:`GRUModule` for constructor arguments and
  examples.
* :func:`set_recurrent_mode` for switching between single-step and recurrent
  execution.
* :func:`set_recurrent_matmul_precision` and
  :func:`get_recurrent_matmul_precision` for Triton precision control.
