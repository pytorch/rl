.. currentmodule:: torchrl.objectives.multiagent

Multi-Agent Objectives
======================

Loss modules for multi-agent reinforcement learning algorithms. These losses
follow the torchrl multi-agent tensordict convention (per-agent tensors
nested under group keys such as ``("agents", "observation")``; see
:class:`~torchrl.envs.VmasEnv` and
:class:`~torchrl.envs.PettingZooEnv`).

MAPPO and IPPO
--------------

:class:`MAPPOLoss` implements Multi-Agent PPO (Yu et al. 2022) — a
decentralised actor paired with a *centralised critic* that conditions on the
joint observation / state. :class:`IPPOLoss` is the independent-learner
counterpart from de Witt et al. 2020: each agent has its own local critic and
there is no centralised information at training time.

Both are thin specialisations of :class:`~torchrl.objectives.ClipPPOLoss`
that:

- default the value estimator to
  :class:`~torchrl.objectives.value.MultiAgentGAE`, which broadcasts
  team-shared rewards / done flags across the agent dimension before
  computing returns;
- default ``normalize_advantage_exclude_dims`` to ``(-2,)`` so the agent dim
  is excluded from advantage standardisation;
- optionally accept a :class:`~torchrl.modules.ValueNorm` subclass — either
  :class:`~torchrl.modules.PopArtValueNorm` (EMA, recommended for drifting
  reward scales) or :class:`~torchrl.modules.RunningValueNorm` (exact
  Welford running stats, recommended for stationary scales) — to stabilise
  the critic loss. The MAPPO paper credits this trick for its strong SMAC
  results.

See ``sota-implementations/multiagent/mappo_ippo.py`` for a hydra-configured
recipe and ``examples/multiagent/mappo_vmas.py`` for a minimal one.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    MAPPOLoss
    IPPOLoss

QMixer
------

:class:`QMixerLoss` mixes local per-agent Q values into a global team Q
value via a learnable mixing network, and trains them jointly with a DQN
update on the global value (Rashid et al. 2018).

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QMixerLoss
