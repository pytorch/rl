.. currentmodule:: torchrl.render

.. _ref_render:

Rendering applications
======================

``torchrl.render`` provides reusable utilities behind the ``rlrender`` and
``torchrl-render`` commands. The command-line entry point imports trusted user
policy and environment factories, loads a local checkpoint, collects one or more
rollouts, captures RGB frames from TensorDict pixels or ``env.render()``, and
writes a reproducible artifact.

Notebook artifacts can also include an optional MuJoCo WASM sidecar viewer. In
that mode, the notebook imports helper functions from ``torchrl.render`` to
start a local Vite viewer, load an MJCF scene in browser-side MuJoCo, and stream
qpos trajectories into the live iframe. When the environment exposes native
MuJoCo state or wraps a Gymnasium MuJoCo environment,
:class:`~torchrl.render.backends.MujocoStateReader` records qpos directly from
the simulator rather than deriving it from policy observations. By default,
``rlrender`` collects and saves trajectories before writing the notebook. Use
``--notebook-rollout-mode live`` to write a notebook that constructs the policy
and environment inside the kernel and generates trajectories when its cells
are run. Use ``--notebook-rollout-mode both`` to save collected rollouts and
also keep an in-notebook collection cell. The generated notebook should stay
thin: reusable display, playback, rollout, and acknowledgement helpers live in
TorchRL rather than being copied into each notebook.

The MuJoCo WASM viewer requires Node.js and either ``npm`` or ``pnpm``. The
viewer installs the generated Vite project's pinned JavaScript dependencies
when ``node_modules`` is absent, which requires network access. The generated
``node_modules`` directory is reused when present.

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
    write_mujoco_wasm_viewer
    display_mujoco_wasm_viewer
    send_mujoco_wasm_qpos
    play_mujoco_wasm_trajectory
    extract_qpos_trajectory

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
    MujocoStateReader
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
