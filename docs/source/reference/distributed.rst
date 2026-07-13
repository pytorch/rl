.. currentmodule:: torchrl.distributed

Native Data Parallelism
=======================

TorchRL provides small, composable primitives for replicated-gradient
training without taking ownership of the training loop.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DataParallelContext

Torchrun setup
--------------

:meth:`DataParallelContext.from_torchrun` reads ``RANK``, ``LOCAL_RANK``, and
``WORLD_SIZE``, selects the local device, and initializes the default process
group when necessary. :meth:`DataParallelContext.from_process_group` instead
wraps an externally owned group and never destroys it.

The explicit replicated-gradient sequence is:

.. code-block:: python

    from torchrl.distributed import DataParallelContext

    context = DataParallelContext.from_torchrun()
    context.broadcast_module(loss_module)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    context.sync_gradients(optimizer)
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm)
    optimizer.step()

``broadcast_module`` synchronizes initial parameters and buffers from rank
zero. ``sync_gradients`` gathers unique optimizer parameters, treats a dense
parameter unused on one rank as a zero gradient on that rank, averages across
the full world size, and leaves parameters unused everywhere with ``grad=None``.
Sparse gradients fail explicitly. Gradient synchronization must run after
backward and before gradient clipping or the optimizer step.

The context exposes ``rank``, ``local_rank``, ``world_size``, ``device``,
``is_rank_zero``, ``barrier()``, idempotent ``close()``, and context-manager
support. Closing a context created by ``from_torchrun`` destroys only a process
group initialized by that context; externally supplied groups remain owned by
the caller.

Ray replay recipe
-----------------

``examples/distributed/replay_buffers/native_data_parallel_dqn.py`` combines
these primitives with a rank-zero-owned Ray replay service, global-batch
rank-aware clients, functional :class:`~torchrl.objectives.DQNLoss`, explicit
gradient averaging, and rank-zero logging. It can be launched with:

.. code-block:: bash

    ray start --head
    torchrun --standalone --nproc-per-node=2 \
        examples/distributed/replay_buffers/native_data_parallel_dqn.py

This first phase covers Ray-owned replay buffers and replicated gradients. It
does not promise mutable forward-buffer synchronization, sparse-gradient
support, checkpoint redistribution, DDP wrapping, FSDP compatibility, or
replicated direct/shared-storage replay sampling.
