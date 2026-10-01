# DreamerEnv

torchrl.envs.model_based.dreamer.DreamerEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/model_based/dreamer.html#DreamerEnv)

Dreamer simulation environment.

This environment is used for imagination rollouts in Dreamer training.
It never terminates (done is always False) since imagination runs for a
fixed horizon. The done-checking methods are overridden to avoid CUDA
synchronization overhead from Python control flow on CUDA tensors.