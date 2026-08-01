.. currentmodule:: torchrl.collectors.distributed

Distributed Collectors
======================

TorchRL provides a set of distributed data collectors. These tools support
multiple backends (``'gloo'``, ``'nccl'``, ``'mpi'`` with the :class:`~.DistributedCollector`
or PyTorch RPC with :class:`~.RPCCollector`) and launchers (``'ray'``,
``submitit`` or ``torch.multiprocessing``).
They can be efficiently used in synchronous or asynchronous mode, on a single
node or across multiple nodes.

*Resources*: Find examples for these collectors in the
`dedicated folder <https://github.com/pytorch/rl/examples/distributed/collectors>`_.

.. note::
  *Choosing the sub-collector*: All distributed collectors support the various single machine collectors.
  One may wonder why using a :class:`~torchrl.collectors.MultiSyncCollector` or a :class:`~torchrl.envs.ParallelEnv`
  instead. In general, multiprocessed collectors have a lower IO footprint than
  parallel environments which need to communicate at each step. Yet, the model specs
  play a role in the opposite direction, since using parallel environments will
  result in a faster execution of the policy (and/or transforms) since these
  operations will be vectorized.

.. note::
  *Choosing the device of a collector (or a parallel environment)*: Sharing data
  among processes is achieved via shared-memory buffers with parallel environment
  and multiprocessed environments executed on CPU. Depending on the capabilities
  of the machine being used, this may be prohibitively slow compared to sharing
  data on GPU which is natively supported by cuda drivers.
  In practice, this means that using the ``device="cpu"`` keyword argument when
  building a parallel environment or collector can result in a slower collection
  than using ``device="cuda"`` when available.

.. note::
  Given the library's many optional dependencies (eg, Gym, Gymnasium, and many others)
  warnings can quickly become quite annoying in multiprocessed / distributed settings.
  By default, TorchRL filters out these warnings in sub-processes. If one still wishes to
  see these warnings, they can be displayed by setting ``torchrl.filter_warnings_subprocess=False``.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DistributedCollector
    RPCCollector
    DistributedSyncCollector
    DistributedDataCollector
    RPCDataCollector
    DistributedSyncDataCollector
    submitit_delayed_launcher
    RayCollector

Legacy names
------------

The following names are kept for backward compatibility:

- ``DistributedDataCollector`` → ``DistributedCollector``
- ``RPCDataCollector`` → ``RPCCollector``
- ``DistributedSyncDataCollector`` → ``DistributedSyncCollector``
