.. currentmodule:: torchrl.render

.. _ref_render:

Rendering applications
======================

``torchrl.render`` provides reusable utilities behind the ``rlrender`` and
``torchrl-render`` commands. The command-line entry point imports trusted user
policy and environment factories, loads a local checkpoint, collects one or more
rollouts, captures RGB frames from TensorDict pixels or ``env.render()``, and
writes a reproducible artifact.

Factories can be addressed as ``module.submodule:callable`` or as a local file
path such as ``/path/to/render_factories.py:make_env``. The base TorchRL package
does not install video or image encoders for this feature. Use the optional
rendering dependencies when writing MP4, GIF, PNG, or YAML-backed configs:

.. code-block:: bash

    uv run --extra rendering rlrender --help

Core API
--------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    render_policy
    make_render_env
    load_render_policy
    collect_render_rollouts
    write_render_artifact
    import_from_string
    call_with_supported_kwargs
    load_checkpoint
    save_render_checkpoint
    checkpoint_hash
    infer_state_dict
    parse_nested_key
    key_to_string
    normalize_policy
    add_step_counter
    seed_env
    normalize_env

Configuration and results
-------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RenderConfig
    RenderEnvSpec
    RenderPolicySpec
    RenderResult
    FrameBundle
    TensorDictPolicyAdapter

Backends
--------

.. currentmodule:: torchrl.render.backends

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RenderBackend
    TensorDictPixelsBackend
    EnvRenderBackend
    NullRenderBackend

Lower-level helpers
-------------------

.. currentmodule:: torchrl.render.cli

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    build_parser
    config_from_args
    main

.. currentmodule:: torchrl.render.notebook

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    build_notebook
    write_render_notebook

.. currentmodule:: torchrl.render.video

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    normalize_frame
    normalize_frame_output
    compose_frame_grid
    encode_video
    encode_gif
    write_png
