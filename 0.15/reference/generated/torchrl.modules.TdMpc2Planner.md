# TdMpc2Planner

*class*torchrl.modules.TdMpc2Planner(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/planners/tdmpc2.html#TdMpc2Planner)

Select actions with the latent-space TD-MPC2 planner.

The planner combines policy-prior trajectories with sampled action
sequences, scores them with the world model and Q ensemble, and refits a
Gaussian distribution to the elite trajectories. Actions are expected to
be normalized to `[-1, 1]`.

The fitted action mean is written under `("next", prev_mean_key)` so a
collector can carry the warm-start state between environment steps.

Parameters:

- **world_model** - TD-MPC2 world model exposing `encoder`, `dynamics`,
and `reward_head` TensorDict modules.
- **policy_prior** - TD-MPC2 policy-prior TensorDict module.
- **q_ensemble** - TD-MPC2 distributional Q ensemble exposing `reduce`.
- **horizon** - Number of imagined action steps. Two additional search
iterations are used when `action_dim >= 20`.
- **discount** - Scalar discount used for imagined rewards and Q bootstrap.
- **num_samples** - Number of candidate action trajectories.
- **num_elites** - Number of candidates used for Gaussian refitting.
- **num_pi_trajs** - Number of fixed policy-prior trajectories.
- **iterations** - Number of elite-refitting iterations.
- **min_std** - Minimum fitted action standard deviation.
- **max_std** - Initial and maximum action standard deviation.
- **temperature** - Elite score temperature.
- **observation_key** - Observation key consumed by the planner.
- **action_key** - Action key written by the planner.
- **is_init_key** - Per-environment reset indicator.
- **prev_mean_key** - Private root key carrying the previous fitted mean.
- **action_dim** - Optional action dimension. It is inferred from the
configured TD-MPC2 policy prior when omitted.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/planners/tdmpc2.html#TdMpc2Planner.forward)

Plan actions for a TensorDict and write the action/state outputs.

Parameters:

**tensordict** - TensorDict containing the observation, optional
`is_init` reset indicator, and optional previous mean.

Returns:

The input TensorDict with the planned action and the fitted mean
under `("next", prev_mean_key)`.

make_tensordict_primer() → [TensorDictPrimer](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer)[[source]](../../_modules/torchrl/modules/planners/tdmpc2.html#TdMpc2Planner.make_tensordict_primer)

Create the primer needed to carry the planner warm-start state.

*property*policy_prior*: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*

Return the live policy prior borrowed from the learner.

*property*q_ensemble*: [TdMpc2QEnsemble](torchrl.modules.TdMpc2QEnsemble.html#torchrl.modules.TdMpc2QEnsemble)*

Return the live Q ensemble borrowed from the learner.

*property*world_model*: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*

Return the live world model borrowed from the learner.