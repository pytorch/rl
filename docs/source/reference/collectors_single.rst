.. currentmodule:: torchrl.collectors

Single Node Collectors
======================

TorchRL provides several collector classes for single-node data collection, each with different execution strategies.

Single node data collectors
---------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DataCollectorBase
    SyncDataCollector
    MultiSyncDataCollector
    MultiaSyncDataCollector
    aSyncDataCollector

Running the Collector Asynchronously
------------------------------------

Passing replay buffers to a collector allows us to start the collection and get rid of the iterative nature of the
collector.
If you want to run a data collector in the background, simply run :meth:`~torchrl.DataCollectorBase.start`:

    >>> collector = SyncDataCollector(..., replay_buffer=rb) # pass your replay buffer
    >>> collector.start()
    >>> # little pause
    >>> time.sleep(10)
    >>> # Start training
    >>> for i in range(optim_steps):
    ...     data = rb.sample()  # Sampling from the replay buffer
    ...     # rest of the training loop

Single-process collectors (:class:`~torchrl.collectors.SyncDataCollector`) will run the process using multithreading,
so be mindful of Python's GIL and related multithreading restrictions.

Multiprocessed collectors will on the other hand let the child processes handle the filling of the buffer on their own,
which truly decouples the data collection and training.

Data collectors that have been started with `start()` should be shut down using
:meth:`~torchrl.DataCollectorBase.async_shutdown`.

.. warning:: Running a collector asynchronously decouples the collection from training, which means that the training
    performance may be drastically different depending on the hardware, load and other factors (although it is generally
    expected to provide significant speed-ups). Make sure you understand how this may affect your algorithm and if it
    is a legitimate thing to do! (For example, on-policy algorithms such as PPO should not be run asynchronously
    unless properly benchmarked).
