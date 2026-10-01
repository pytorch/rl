# IsaacGymWrapper

torchrl.envs.IsaacGymWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/isaacgym.html#IsaacGymWrapper)

Wrapper for IsaacGymEnvs environments.

The original library can be found [here](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
and is based on IsaacGym which can be downloaded through NVIDIA's webpage <https://developer.nvidia.com/isaac-gym>_.

Note

IsaacGym environments cannot be executed consecutively, ie. instantiating one
environment after another (even if it has been cleared) will cause
CUDA memory issues. We recommend creating one environment per process only.
If you need more than one environment, the best way to achieve that is
to spawn them across processes.

Note

IsaacGym works on CUDA devices by essence. Make sure your machine
has GPUs available and the required setup for IsaacGym (eg, Ubuntu 20.04).