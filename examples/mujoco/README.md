# MuJoCo examples

## MicroDuck PPO

[`ppo_microduck.py`](ppo_microduck.py) defines a stand/balance task once across
the native MuJoCo, MJX, and `mujoco-torch` backends. The accompanying
[`microduck_ppo.ipynb`](microduck_ppo.ipynb) trains the task locally and renders
the policy with native MuJoCo or the interactive MuJoCo WASM viewer.

[Watch the local PPO before/after rollout](assets/microduck_ppo_before_after.mp4).
The synchronized video uses native MuJoCo on CPU, an identical `STAND` reset,
and deterministic actions. The initial policy terminates at step 108; the PPO
checkpoint terminates at step 143.

This is a feasibility run rather than a benchmark. Training used eight
environments, 240 iterations of 128 steps, two PPO epochs, 256-sample
minibatches, a linearly decayed `1e-4` learning rate, an entropy coefficient of
`1e-4`, and a critic coefficient of `0.5`. Evaluation every 20 iterations
selected iteration 60. Across eight 500-step evaluation streams, mean step
reward changed from 2.740 to 2.801 and terminations from 32 to 24.
