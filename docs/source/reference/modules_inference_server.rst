.. currentmodule:: torchrl.modules.inference_server

Inference Server
================

.. _ref_inference_server:

The inference server provides auto-batching model serving for RL actors.
Multiple actors submit individual TensorDicts; the server transparently
batches them, runs a single model forward pass, and routes results back.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    InferenceServer
    InferenceClient
    InferenceTransport
