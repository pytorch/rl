.. currentmodule:: torchrl.data

.. _TED-format:

Datasets
========

TorchRL provides dataset utilities for offline RL and data management.

TorchRL Episode Data (TED) Format
---------------------------------

The TED format is TorchRL's standard data layout for offline RL datasets. It structures
trajectories as nested tensors where each element contains the full trajectory data,
making it efficient for sequence-based sampling and training.


Dataset loading registry
------------------------

Many offline datasets can be constructed from a compact string identifier with
:func:`~torchrl.data.datasets.load_dataset`. The identifier is split as
``"<source>:<dataset-id>"``. The ``source`` prefix selects a registered dataset
factory and the remainder is forwarded as the first constructor argument, with
any extra keyword arguments forwarded unchanged.

TorchRL registers the built-in dataset families at import time, including
``"atari"``, ``"atari_dqn"``, ``"d4rl"``, ``"gen_dgrl"``, ``"lerobot"``,
``"minari"``, ``"openml"``, ``"openx"``, ``"roboset"``, and ``"vd4rl"``.
For example:

.. code-block:: python

    from torchrl.data.datasets import load_dataset

    dataset = load_dataset("d4rl:halfcheetah-medium-v2", batch_size=256)
    minari_dataset = load_dataset(
        "minari:mujoco/hopper/expert-v0",
        batch_size=256,
        split_trajs=True,
    )

Projects can add their own sources with
:func:`~torchrl.data.datasets.register_dataset`. A registered factory can be a
callable or a lazy import string of the form ``"module:attribute"``. Factories
are called as ``factory(dataset_id, **kwargs)``.

.. code-block:: python

    from torchrl.data.datasets import load_dataset, register_dataset

    class MyDataset:
        def __init__(self, dataset_id, *, batch_size):
            self.dataset_id = dataset_id
            self.batch_size = batch_size

    register_dataset("my_backend", MyDataset)
    dataset = load_dataset("my_backend:my-task-v0", batch_size=128)

    register_dataset(
        "my_lazy_backend",
        "my_package.datasets:MyDataset",
    )

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    datasets.AtariDQNExperienceReplay
    datasets.D4RLExperienceReplay
    datasets.GenDGRLExperienceReplay
    datasets.LeRobotExperienceReplay
    datasets.MinariExperienceReplay
    datasets.OpenMLExperienceReplay
    datasets.OpenXExperienceReplay
    datasets.RobosetExperienceReplay
    datasets.VD4RLExperienceReplay

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    datasets.lerobot_columns_to_tensordict

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    datasets.load_dataset
    datasets.register_dataset
