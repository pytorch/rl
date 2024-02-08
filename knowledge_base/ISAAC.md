# Isaac and IsaacGym

Isaac Gym is NVIDIA’s prototype physics simulation environment
for end-to-end GPU accelerated reinforcement learning research.

## Installation instructions

Got to https://developer.nvidia.com/isaac-gym and register using using the link
at the bottom of the page. 

Download the `IsaacGym_Preview_4_Package.tar.gz` file and expand it on the machine
that you want to use:
```shell
tar -xf IsaacGym_Preview_4_Package.tar.gz isaacgym/
cd isaacgym
```

Create your pre-built conda environment (this may take a while!)

```shell
./create_conda_env_rlgpu.sh
```

