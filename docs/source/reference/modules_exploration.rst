.. currentmodule:: torchrl.modules

Exploration Strategies
======================

Exploration modules add noise to actions to enable exploration during training.
:class:`~torchrl.modules.NoisyLinear` instead injects learnable noise in
parameter space. :meth:`~torchrl.modules.NoisyLinear.reset_noise` is called
only at initialization, so callers must resample (``module.apply(reset_noise)``);
:func:`~torchrl.trainers.helpers.make_trainer` already does this on the
``pre_optim_steps`` hook when ``cfg.noisy`` is set.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    AdditiveGaussianModule
    ConsistentDropoutModule
    EGreedyModule
    NoisyLazyLinear
    NoisyLinear
    OrnsteinUhlenbeckProcessModule

Helpers
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    reset_noise
    set_exploration_modules_spec_from_env
