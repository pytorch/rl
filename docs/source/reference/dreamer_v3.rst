DreamerV3 in a nutshell
=======================

`DreamerV3 <https://arxiv.org/abs/2301.04104>`_ is a model-based reinforcement
learning algorithm. It learns a compact model of the environment from replayed
experience, then trains an actor and a critic on trajectories generated inside
that model. The real environment supplies data for the world model; most policy
improvement happens in latent-space imagination.

Paper and maintained implementation
-----------------------------------

This page treats the `DreamerV3 paper <https://arxiv.org/abs/2301.04104>`_ as
the source of truth for the algorithm. The author-maintained
`JAX implementation <https://github.com/danijar/dreamerv3>`_ continues to
evolve and its named experiment presets can differ from the protocol reported
in the paper. TorchRL documents those presets as separate reproduction targets
rather than redefining the paper algorithm around the latest JAX configuration.

Some constructor defaults predate full paper parity and remain for backward
compatibility. The runnable DreamerV3 recipes pass the paper-compatible loss
settings explicitly; changes to public defaults require the normal deprecation
cycle.

The high-level data flow is:

.. code-block:: text

    real transition sequences
              |
              v
    encoder -> posterior RSSM state -> reconstruction, reward, continuation
                   ^       |
                   |       v
             prior dynamics + action
                           |
                           v
                 imagined trajectories
                           |
                           v
                  actor + online critic
                           |
                           v
                 slow (target) critic

Nomenclature
------------

Dreamer papers and implementations use several names for closely related
objects. In the TorchRL API:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Term
     - Meaning
   * - **World model**
     - The observation encoder, recurrent state-space model (RSSM), observation
       decoder, reward predictor, and optional continuation predictor.
   * - **Belief** or **deterministic state** (``h_t``)
     - The recurrent hidden state that summarizes history. TorchRL stores it
       under the ``"belief"`` key.
   * - **State** or **stochastic state** (``z_t``)
     - A sample from the RSSM's categorical latent variables. TorchRL stores
       the flattened straight-through one-hot sample under ``"state"``.
   * - **Prior**, **dynamics**, or **transition model**
     - Predicts the next categorical state from the previous state, belief, and
       action, without seeing the next observation. See
       :class:`~torchrl.modules.models.RSSMPriorV3`.
   * - **Posterior** or **representation model**
     - Corrects the prior using the encoded next observation. It is used while
       learning from real sequences. See
       :class:`~torchrl.modules.models.RSSMPosteriorV3`.
   * - **Imagination**
     - A latent rollout that uses the prior, reward model, continuation model,
       and actor, but no real observations.
   * - **Critic** or **value model**
     - The online network that predicts lambda returns from the RSSM state and
       belief.
   * - **Slow critic**, **target critic**, or **EMA critic**
     - A lagged copy of the online critic. It is updated by Polyak averaging
       and provides a stable auxiliary target for critic regularization.
       "Slow" refers to its parameter updates, not its optimizer or runtime.
   * - **Continuation**
     - The learned probability that an imagined trajectory continues. It
       replaces a fixed survival assumption when weighting returns and losses.

How the RSSM works
------------------

The RSSM splits its latent representation into a deterministic recurrent state
and a stochastic categorical state. At each real-data time step:

1. :class:`~torchrl.modules.models.RSSMPriorV3` updates the belief from the previous
   stochastic state and action, then predicts prior categorical logits.
2. :class:`~torchrl.modules.models.RSSMPosteriorV3` combines that belief with the
   encoded observation and predicts posterior categorical logits.
3. Both networks sample hard categorical states with a straight-through
   gradient estimator. ``unimix`` can mix a small uniform component into the
   probabilities to prevent overconfident categories.
4. :class:`~torchrl.modules.models.RSSMRolloutV3` carries the posterior state and
   belief through a sequence and resets them at episode boundaries.

During imagination there is no observation, so only the prior advances the
latent state. The actor and prediction heads consume both ``state`` and
``belief``. ``RSSMPriorV3`` supports a conventional GRU and the grouped
``"block_gru"`` core used by the full DreamerV3 example. The accompanying
:class:`~torchrl.modules.DreamerV3MLP` provides the RMS-normalized SiLU MLP
blocks used by the example's encoder, decoder, actor, critic, and prediction
heads.

For recurrent features outside an RSSM, :class:`~torchrl.modules.DreamerV3BlockGRUCell`
exposes the same block-diagonal update as a single-step module, while
:class:`~torchrl.modules.DreamerV3BlockGRU` executes batch-major sequences with
mixed episode resets. The sequence module uses an ordinary-autograd reference
loop by default and offers an opt-in compiled scan backend for long training
sequences. On CUDA, the explicit ``"triton"`` backend supports SiLU, Tanh, and
ReLU dynamics and fuses the complete forward and reverse-time recurrences. It
runs in ``float32`` or ``bfloat16`` (mixed input and hidden dtypes are promoted
like the reference backend; other dtypes raise an error), requires Triton 3.3
or newer, and never silently falls back to another backend.

Selecting and benchmarking the sequence backend
-----------------------------------------------

The sequence backend is selected directly on the high-level module:

.. code-block:: python

    from torchrl.modules import DreamerV3BlockGRU

    gru = DreamerV3BlockGRU(
        input_size=512,
        hidden_size=512,
        recurrent_backend="triton",
    ).cuda()

The reference backend remains the portable default. Select ``"scan"`` or
``"triton"`` explicitly so missing dependencies or unsupported devices are
reported instead of silently changing execution.

Installed TorchRL packages also provide ``torchrl-benchmark-rnn``. The
following command compares the optimized DreamerV3 sequence backends on the
current CUDA device using synchronized forward and backward timings, peak
memory, and 95% confidence intervals:

.. code-block:: bash

    torchrl-benchmark-rnn --rnn block_gru \
        --backends scan,triton --batches 16 --seq-lens 64,512 \
        --hiddens 512 --input-size 512 --projection-size 512 --blocks 8 \
        --dtype bfloat16 --warmup 10 --iters 30

Use the batch size, sequence length, widths, block count, dtype, and compile
modes from the intended workload: backend performance is hardware- and
shape-dependent.

The three objectives
--------------------

World model
~~~~~~~~~~~

:class:`~torchrl.objectives.DreamerV3ModelLoss` trains the model on real
transition sequences. Its components are:

* a dynamics KL that trains the prior toward a stopped-gradient posterior;
* a representation KL that trains the posterior toward a stopped-gradient
  prior;
* free nats and optional uniform mixing for the categorical distributions;
* an L1 or L2 reconstruction loss in symlog space;
* a reward loss using symlog-spaced two-hot bins, or symlog MSE; and
* an optional binary continuation loss.

``kl_mode="separate"`` exposes the dynamics and representation KL terms
separately, as used by the full example. ``kl_mode="balanced"`` provides the
combined balanced-KL form.

Actor
~~~~~

:class:`~torchrl.objectives.DreamerV3ActorLoss` starts from posterior states
produced by the world model and rolls the actor through a
:class:`~torchrl.envs.model_based.dreamer.DreamerEnv`. It computes lambda
returns from predicted rewards, values, and optional continuation
probabilities. It supports:

* TD(0), TD(1), and TD(lambda) return estimators;
* REINFORCE with a stopped-gradient advantage, or reparameterization gradients
  for suitable continuous policies;
* an entropy bonus;
* cumulative discount/continuation weighting; and
* EMA percentile-range normalization of REINFORCE returns.

Critic and slow critic
~~~~~~~~~~~~~~~~~~~~~~

:class:`~torchrl.objectives.DreamerV3ValueLoss` fits the online critic to the
lambda returns produced by the actor loss. The critic can use symlog MSE or a
distributional two-hot cross-entropy loss.

Setting ``slow_critic_regularization`` to a positive value creates target
critic parameters inside the value loss. The slow critic is a soft-updated
copy of the online critic:

.. math::

    \theta_{\mathrm{slow}} \leftarrow
    (1 - \tau)\,\theta_{\mathrm{slow}} +
    \tau\,\theta_{\mathrm{online}}.

The slow critic's stopped-gradient prediction is an additional target for the
online critic. In the current TorchRL objective, the online critic still
provides the bootstrap values used to form imagined lambda returns; the slow
critic regularizes critic learning rather than replacing that bootstrap.

Target updates are deliberately external to the loss. Associate a
:class:`~torchrl.objectives.SoftUpdate` with the value loss and call it after
each critic optimizer step:

.. code-block:: python

    from torchrl.objectives import DreamerV3ValueLoss
    from torchrl.objectives.utils import SoftUpdate

    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        actor_loss=actor_loss,
        slow_critic_regularization=1.0,
    )
    slow_critic_updater = SoftUpdate(value_loss, tau=0.02)

    # After loss.backward() and optimizer.step():
    slow_critic_updater.step()

Replay critic loss
~~~~~~~~~~~~~~~~~~

The reference implementation also fits the critic on the real replay sequences,
not only on imagined trajectories.
:meth:`~torchrl.objectives.DreamerV3ValueLoss.replay_value_loss` computes that
term. Its return at each replay state uses the following replay reward and
bootstraps from the first imagined lambda return of the next state, so the
critic is fitted on real replay states as well as imagined states. The method
reads its
``reward``, ``done``, ``terminated`` and ``bootstrap`` entries through
:attr:`~torchrl.objectives.DreamerV3ValueLoss.tensor_keys`, so
:meth:`~torchrl.objectives.LossModule.set_keys` can redirect them:

.. code-block:: python

    value_loss.set_keys(bootstrap="first_imagined_return")
    replay_td = value_loss.replay_value_loss(replay_features)
    loss = replay_td["loss_replay_value"]

Because the input features stay attached, this term also trains the RSSM
representation when the world-model loss returns live features.

Optimization and training loop
------------------------------

The loss modules do not create optimizers. This keeps optimizer ownership and
the update schedule explicit. A typical update cycle is:

1. Sample contiguous real transition sequences from replay.
2. Update the world model on KL, reconstruction, reward, and continuation
   losses.
3. Detach posterior states from the real sequence and use them as imagination
   starting points.
4. Update the actor on imagined lambda returns.
5. Update the online critic on those same detached returns.
6. Soft-update the slow critic.

The runnable ``sota-implementations/dreamer_v3`` example uses separate Adam
optimizers for the world model, actor, and critic. They share a learning rate,
Adam coefficients, linear learning-rate warmup, and adaptive gradient clipping.
Those choices belong to the training recipe rather than the loss API, so users
can substitute another optimizer or schedule without changing the objectives.

API map
-------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Component
     - Purpose
   * - :class:`~torchrl.modules.models.RSSMPriorV3`
     - Categorical latent dynamics and deterministic recurrent update.
   * - :class:`~torchrl.modules.models.RSSMPosteriorV3`
     - Observation-conditioned categorical representation model.
   * - :class:`~torchrl.modules.models.RSSMRolloutV3`
     - Sequential prior/posterior filtering over replayed trajectories.
   * - :class:`~torchrl.modules.DreamerV3MLP`
     - RMS-normalized MLP building block.
   * - :class:`~torchrl.modules.SymExpTwoHot`
     - Symlog-spaced categorical scalar encoder, decoder, and loss helper.
   * - :class:`~torchrl.objectives.DreamerV3ModelLoss`
     - World-model objective.
   * - :class:`~torchrl.objectives.DreamerV3ActorLoss`
     - Latent-imagination actor objective and lambda-return construction.
   * - :class:`~torchrl.objectives.DreamerV3ValueLoss`
     - Online and slow-critic objective.
   * - :class:`~torchrl.objectives.SoftUpdate`
     - External Polyak update for the slow critic.
   * - :func:`~torchrl.objectives.symlog`,
       :func:`~torchrl.objectives.symexp`, and two-hot helpers
     - Scale-robust scalar transformations for custom heads and losses.

For a complete training setup, see the
`DreamerV3 example <https://github.com/pytorch/rl/tree/main/sota-implementations/dreamer_v3>`_.
