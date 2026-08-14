.. _value_transforms:

Value transforms in actor-critic objectives
===========================================

Value transforms compress large signed returns into a better-conditioned
critic prediction space without changing the units used by the reinforcement
learning algorithm. If ``h`` is the transform, the critic predicts
``y = h(V)``. A one-step target is assembled as

.. math::

   z = r + \gamma (1 - d) h^{-1}(y_{next}),

and critic regression compares ``y`` with ``h(z)``. The inverse is therefore
applied *before* minima, entropy terms, Bellman arithmetic, return estimation,
advantage estimation, and actor objectives. The completed target is transformed
once, immediately before critic regression.

Space contract
--------------

.. list-table:: TensorDict and diagnostic spaces
   :header-rows: 1

   * - Value
     - Space
   * - ``state_value`` and ``state_action_value``
     - Transformed critic prediction space
   * - ``reward``, ``advantage``, and ``value_target``
     - Raw reward-return space
   * - Explicit estimator ``next_value=``
     - Raw value space
   * - PPO ``clip_value``
     - Raw value units
   * - ``pred_value``, ``target_value``, and ``next_state_value`` diagnostics
     - Raw value space
   * - Replay ``td_error``
     - Existing absolute or squared convention, using the transformed residual

:class:`~torchrl.modules.ValueOperator` deliberately does not apply a value
transform. Its wrapped module must emit transformed predictions. The loss owns
the :class:`~torchrl.modules.ValueTransform` and passes the exact same object to
lazy value estimators. A prebuilt estimator must also hold that same object;
using an equivalent but distinct transform raises an error so the two sides
cannot silently diverge.

DDPG and TD3
------------

The following critic emits symlog predictions. Both DDPG and TD3 invert those
predictions for the actor objective and target bootstrap, then transform the
finished Bellman target for regression.

.. code-block:: python

    import torch
    from tensordict import TensorDict
    from torch import nn
    from torchrl.data import Bounded
    from torchrl.modules import Actor, SymLogValueTransform, ValueOperator
    from torchrl.objectives import DDPGLoss, TD3Loss

    obs_dim, action_dim = 3, 2
    action_spec = Bounded(-torch.ones(action_dim), torch.ones(action_dim))

    def make_actor():
        return Actor(nn.Linear(obs_dim, action_dim), spec=action_spec)

    class TransformedQ(nn.Module):
        def __init__(self, transform):
            super().__init__()
            self.linear = nn.Linear(obs_dim + action_dim, 1)
            self.transform = transform

        def forward(self, observation, action):
            raw_q = self.linear(torch.cat((observation, action), -1))
            return self.transform(raw_q)

    ddpg_transform = SymLogValueTransform()
    ddpg_q = ValueOperator(
        TransformedQ(ddpg_transform),
        in_keys=["observation", "action"],
    )
    ddpg_loss = DDPGLoss(
        make_actor(), ddpg_q, value_transform=ddpg_transform
    )

    td3_transform = SymLogValueTransform()
    td3_q = ValueOperator(
        TransformedQ(td3_transform),
        in_keys=["observation", "action"],
    )
    td3_loss = TD3Loss(
        make_actor(),
        td3_q,
        action_spec=action_spec,
        value_transform=td3_transform,
    )

    replay_sample = TensorDict(
        {
            "observation": torch.randn(8, obs_dim),
            "action": action_spec.rand((8,)),
            "next": {
                "observation": torch.randn(8, obs_dim),
                "reward": torch.randn(8, 1) * 100.0,
                "done": torch.zeros(8, 1, dtype=torch.bool),
                "terminated": torch.zeros(8, 1, dtype=torch.bool),
            },
        },
        [8],
    )
    ddpg_output = ddpg_loss(replay_sample.clone())
    td3_output = td3_loss(replay_sample.clone())

Continuous SAC v1 and v2
------------------------

For SAC, ``Q - alpha * log_prob`` is computed in raw units. SAC v1 requires
both its Q and V networks to emit predictions in the same transform space; SAC
v2 only needs the transformed Q critic.

.. code-block:: python

    from tensordict.nn import NormalParamExtractor, TensorDictModule
    from torchrl.modules import ProbabilisticActor, TanhNormal
    from torchrl.objectives import SACLoss

    def make_stochastic_actor():
        policy_module = TensorDictModule(
            nn.Sequential(
                nn.Linear(obs_dim, 2 * action_dim),
                NormalParamExtractor(),
            ),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        return ProbabilisticActor(
            policy_module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            spec=action_spec,
            return_log_prob=True,
        )

    sac_transform = SymLogValueTransform()
    sac_q = ValueOperator(
        TransformedQ(sac_transform),
        in_keys=["observation", "action"],
    )
    sac_v = ValueOperator(
        nn.Sequential(nn.Linear(obs_dim, 1), sac_transform),
        in_keys=["observation"],
    )
    sac_v1_loss = SACLoss(
        make_stochastic_actor(),
        sac_q,
        sac_v,
        value_transform=sac_transform,
    )

    sac_v2_transform = SymLogValueTransform()
    sac_v2_q = ValueOperator(
        TransformedQ(sac_v2_transform),
        in_keys=["observation", "action"],
    )
    sac_v2_loss = SACLoss(
        make_stochastic_actor(),
        sac_v2_q,
        value_transform=sac_v2_transform,
    )
    sac_v1_output = sac_v1_loss(replay_sample.clone())
    sac_v2_output = sac_v2_loss(replay_sample.clone())

PPO, ClipPPO, and KLPENPPO
--------------------------

PPO rollout values remain transformed in ``state_value``. GAE inverts them to
produce raw ``advantage`` and ``value_target`` entries. Pass the same transform
object to GAE and the PPO loss. Value clipping is performed around the old and
current *raw* predictions; only the clipped candidate is transformed for the
critic loss.

.. code-block:: python

    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    ppo_transform = SymLogValueTransform()
    critic = ValueOperator(
        nn.Sequential(nn.Linear(obs_dim, 1), ppo_transform),
        in_keys=["observation"],
    )
    gae = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        value_transform=ppo_transform,
    )
    ppo_actor = make_stochastic_actor()
    ppo_loss = ClipPPOLoss(
        ppo_actor,
        critic,
        clip_value=0.2,  # raw return units
        value_transform=ppo_transform,
    )

    rollout = TensorDict(
        {
            "observation": torch.randn(2, 8, obs_dim),
            "next": {
                "observation": torch.randn(2, 8, obs_dim),
                "reward": torch.randn(2, 8, 1) * 100.0,
                "done": torch.zeros(2, 8, 1, dtype=torch.bool),
                "terminated": torch.zeros(2, 8, 1, dtype=torch.bool),
            },
        },
        [2, 8],
        names=[None, "time"],
    )
    with torch.no_grad():
        ppo_actor(rollout)
    gae(rollout)
    losses = ppo_loss(rollout)

The same setup applies to :class:`~torchrl.objectives.PPOLoss` and
:class:`~torchrl.objectives.KLPENPPOLoss`.

Target networks, priorities, and logging
----------------------------------------

Target critics emit transformed predictions just like online critics. Target
minima and SAC entropy subtraction happen after inversion. Replay priorities
keep each objective's historical convention (squared for DDPG, TD3, and SAC
v1; absolute for SAC v2), but measure the residual between the transformed
prediction and transformed target. Pass ``priority_function`` to these losses
to override that convention with a callable accepting the prediction and target
in transformed space and returning a per-element priority, for example
``priority_function=lambda prediction, target: (prediction - target).abs()``.
Values intended for monitoring are converted back to raw units so reward-scale
dashboards remain interpretable.

Hydra
-----

All supported loss and GAE configs accept nested ``_target_`` transforms. For
example:

.. code-block:: yaml

    loss_module:
      _target_: torchrl.trainers.algorithms.configs.objectives._make_ppo_loss
      actor_network: ${actor}
      critic_network: ${critic}
      loss_type: clip
      clip_value: 0.2
      value_transform:
        _target_: torchrl.modules.SymLogValueTransform

    value:
      _target_: torchrl.objectives.value.GAE
      gamma: 0.99
      lmbda: 0.95
      value_network: ${critic}
      value_transform: ${loss_module.value_transform}

Unsupported combinations
------------------------

Value transforms are currently supported for DDPG, TD3, continuous SAC v1/v2,
PPO, ClipPPO, KLPENPPO, TD(0), TD(1), TD(lambda), GAE, MultiAgentGAE, and
V-trace. They are not scalar wrappers for distributional, categorical, or
two-hot critics, and are not supported by Discrete SAC. Do not combine a value
transform with MAPPO/IPPO :class:`~torchrl.modules.ValueNorm` or PopArt: those
mechanisms each define a different prediction-space normalization, and TorchRL
rejects configuring both at once.

Passing ``None`` selects the existing identity path without adding tensor
operations. A transformed path performs the inverse only when a prediction is
consumed and applies one forward transform to each completed regression target.

.. seealso::
    :class:`~torchrl.modules.ValueTransform`,
    :class:`~torchrl.modules.SymLogValueTransform`,
    :class:`~torchrl.modules.SignedHyperbolicValueTransform`,
    :class:`~torchrl.modules.ValueOperator`, and
    :class:`~torchrl.objectives.value.ValueEstimatorBase`.
