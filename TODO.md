# TODO

## Bugfix:
- deal with noisy layers in target network updating (buffers? copying?)

## Logging features
- total reward
- video
- tensorboard API

## Profiling
- Profiling API

## Features / Modules
- Distributional RL
- N-step return
- TensorDicts:
  - view
  - reshape
  - masking
  - permutation


## Cleaning
- Probabilistic operator and Distribution interface (in case of exploration)
- TensorDict batch_size should be a common feature of all tensor_dicts
- Wording: terms are important but can be confusing. A strong line should be drawn between:
- agent, policy, actor
- trajectory, rollout
- Envs should have separate `reset()` and `seed()` methods, not a `reset(seed=seed)` that has a double responsability.
- tell policy to compute log_probs, not collector
- move `env.rollout()` out of `env` classes and make it a function.
- deprecate env dtype

## Optional / TBD
- Abstract tensor_dict class?

## Challenges
- having the possibility to reset a policy: integrate policy and env (like in torch_agent)?
  - Usage: policies with OU noise
  - Alternative: store a tensor in tensor_dict that is zeroed at reset
