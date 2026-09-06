.. currentmodule:: torchrl.objectives

Policy Gradient Methods
=======================

Loss modules for policy gradient algorithms.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    PPOLoss
    ClipPPOLoss
    KLPENPPOLoss
    A2CLoss
    ReinforceLoss

Decoupled proximal policy (PPO-EWMA)
------------------------------------

PPO uses the policy that collected the data (the ``sample_log_prob`` entry)
for two distinct purposes: as the denominator of the importance-sampling
ratio, and as the reference the trust region (clipping or KL penalty) pulls
the updated policy towards. `Hilton et al. (2021)
<https://arxiv.org/abs/2110.00641>`_ observe that only the first role requires
the *behavior* policy; the second can use any recent *proximal* policy.

Passing ``delay_actor=True`` to :class:`ClipPPOLoss` or :class:`KLPENPPOLoss`
keeps a detached copy of the actor parameters under
``target_actor_network_params`` and uses it as the proximal policy: the ratio
being clipped (or the KL reference) becomes ``pi_theta / pi_prox`` and the
surrogate is re-weighted by ``pi_prox / pi_behav`` so that the gradient
estimate stays unbiased for the data at hand. Stepping a
:class:`~torchrl.objectives.SoftUpdate` after every optimizer step turns the
proximal policy into an exponentially-weighted moving average of the policy,
which is PPO-EWMA; a :class:`~torchrl.objectives.HardUpdate` freezes it for a
fixed number of steps instead. ``max_importance_ratio`` caps the behavior
ratio for numerical stability with stale data. This decouples the strength of
the trust region from how, and how recently, the data was collected.

    >>> loss_module = ClipPPOLoss(actor, critic, delay_actor=True, max_importance_ratio=100.0)
    >>> updater = SoftUpdate(loss_module, eps=0.889)  # eps is the EWMA decay rate beta_prox
    >>> for batch in replay_buffer:
    ...     losses = loss_module(batch)
    ...     loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    ...     updater.step()
