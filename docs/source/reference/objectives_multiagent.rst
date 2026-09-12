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

COMA
----

:class:`COMALoss` implements Counterfactual Multi-Agent Policy Gradients
(Foerster et al. 2018) — a decentralised actor paired with a *centralised*
critic that outputs one action value per possible action of the acting
agent. The actor's advantage is a counterfactual baseline: the critic's own
output is marginalised over the acting agent's action (holding every other
agent's action fixed), crediting an agent only for the part of the outcome
its own action choice affected.

The critic is centralised through its inputs rather than through a mixing
network: it is built with the acting agent's observation together with
either the other agents' actions
(:func:`~torchrl.objectives.multiagent.coma.add_action_without_self`) or a
joint/global state
(:func:`~torchrl.objectives.multiagent.coma.add_joint_observation`,
:func:`~torchrl.objectives.multiagent.coma.add_masked_joint_action`). The
critic itself is trained with an n-step TD target bootstrapped from a target
network via :meth:`~torchrl.objectives.multiagent.COMALoss.compute_value_target`.

See ``sota-implementations/multiagent/coma.py`` for a hydra-configured recipe.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    COMALoss
