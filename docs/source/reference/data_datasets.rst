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

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    datasets.AtariDQNExperienceReplay
    datasets.D4RLExperienceReplay
    datasets.GenDGRLExperienceReplay
    datasets.MinariExperienceReplay
    datasets.OpenMLExperienceReplay
    datasets.OpenXExperienceReplay
    datasets.RobosetExperienceReplay
    datasets.TDMPC2ExperienceReplay
    datasets.VD4RLExperienceReplay
