.. currentmodule:: torchrl.modules.inference_server

Inference Server
================

.. _ref_inference_server:

The inference server provides auto-batching model serving for RL actors.
Multiple actors submit individual TensorDicts; the server transparently
batches them, runs a single model forward pass, and routes results back.

Core API
--------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    InferenceServer
    InferenceClient
    InferenceTransport

Transport Backends
------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ThreadingTransport
    MPTransport
    RayTransport
    MonarchTransport

Usage
-----

The simplest setup uses :class:`ThreadingTransport` for actors that are
threads in the same process:

.. code-block:: python

    from tensordict.nn import TensorDictModule
    from torchrl.modules.inference_server import (
        InferenceServer,
        ThreadingTransport,
    )
    import torch.nn as nn
    import concurrent.futures

    policy = TensorDictModule(
        nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4)),
        in_keys=["observation"],
        out_keys=["action"],
    )

    transport = ThreadingTransport()
    server = InferenceServer(policy, transport, max_batch_size=32)
    server.start()
    client = transport.client()

    # actor threads call client(td) -- batched automatically
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        ...

    server.shutdown()

Weight Synchronisation
^^^^^^^^^^^^^^^^^^^^^^

The server integrates with :class:`~torchrl.weight_update.WeightSyncScheme`
to receive updated model weights from a trainer between inference batches:

.. code-block:: python

    from torchrl.weight_update import SharedMemWeightSyncScheme

    weight_sync = SharedMemWeightSyncScheme()
    # Initialise on the trainer (sender) side first
    weight_sync.init_on_sender(model=training_model, ...)

    server = InferenceServer(
        model=inference_model,
        transport=ThreadingTransport(),
        weight_sync=weight_sync,
    )
    server.start()

    # Training loop
    for batch in dataloader:
        loss = loss_fn(training_model(batch))
        loss.backward()
        optimizer.step()
        weight_sync.send(model=training_model)  # pushed to server
