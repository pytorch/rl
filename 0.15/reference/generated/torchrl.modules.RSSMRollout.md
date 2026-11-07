# RSSMRollout

*class*torchrl.modules.RSSMRollout(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMRollout)

Rollout the RSSM network.

Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
states and beliefs.
The previous posterior is used as the prior for the next time step.
The forward method returns a stack of all intermediate states and beliefs.

Reference: [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)

Parameters:

- **rssm_prior** (*TensorDictModule*) - Prior network.
- **rssm_posterior** (*TensorDictModule*) - Posterior network.
- **use_scan** (*bool**,**optional*) - If True, uses torch._higher_order_ops.scan for
the rollout loop. This is more torch.compile friendly but may have
different performance characteristics. Defaults to False.
- **compile_step** (*bool**,**optional*) - If True, compiles the individual step function.
Only used when use_scan=False. Defaults to False.
- **compile_backend** (*str**,**optional*) - Backend to use for compilation.
Defaults to "inductor".
- **compile_mode** (*str**,**optional*) - Mode to use for compilation.
Defaults to None (uses PyTorch default).

forward(*tensordict*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMRollout.forward)

Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

The rollout requires a belief and posterior state primer.

At each step, two probability distributions are built and sampled from:

- A prior distribution p(s_{t+1} | s_t, a_t, b_t) where b_t is a

deterministic transform of the form b_t(s_{t-1}, a_{t-1}). The
previous state s_t is sampled according to the posterior
distribution (see below), creating a chain of posterior-to-priors
that accumulates evidence to compute a prior distribution over
the current event distribution:
p(s_{t+1} s_t | o_t, a_t, s_{t-1}, a_{t-1}) = p(s_{t+1} | s_t, a_t, b_t) q(s_t | b_t, o_t)
- A posterior distribution of the form q(s_{t+1} | b_{t+1}, o_{t+1})

which amends to q(s_{t+1} | s_t, a_t, o_{t+1})