# Working with [`habitat-lab`](https://github.com/facebookresearch/habitat-lab)

## Setting up your environment for habitat and torchrl

### Installing habitat-lab from pip

Instructions can be found on [habitat github repo](https://github.com/facebookresearch/habitat-lab).

  1. **Preparing conda env**

     Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
     ```bash
     export MY_TEST_ENV=habitat_test
     conda deactivate
     conda env remove -n $MY_TEST_ENV -y
     conda create -n $MY_TEST_ENV python=3.7 cmake=3.14.0 -y
     conda activate $MY_TEST_ENV
     ```

  2. **conda install habitat-sim**
     To install habitat-sim with bullet physics and in headless mode (usually necessary to run habitat on a cluster)
     ```bash
     conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
     pip install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-lab
     
     # This is to reduce verbosity
     export MAGNUM_LOG=quiet && export HABITAT_SIM_LOG=quiet
     ```
     If you don't want to install it in headless mode, simply remove the `headless` package from the `conda install` command.

     See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

  3. **Install habitat-baselines**.

      The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
       pip install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-baselines
      ```
### Installing TorchRL

Follow the instructions on the [README.md](../README.md).

### Using Habitat
To get the list of available Habitat envs, simply run the following command:
```python
from torchrl.envs.libs.habitat import HabitatEnv, _has_habitat
assert _has_habitat  # checks that habitat is installed
print([_env for _env in HabitatEnv.available_envs if _env.startswith("Habitat")])
```

## Common issues


1. `ImportError: /usr/lib/x86_64-linux-gnu/libOpenGL.so.0: undefined symbol: _glapi_tls_Current`
  **Solution**: as in [MUJOCO]([url](https://github.com/pytorch/rl/blob/main/knowledge_base/MUJOCO_INSTALLATION.md)) debug, Link conda to the right libOpenGL.so file (replace /path/to/conda and mujoco_env with the proper paths and names):
  ```shell
  conda install -y -c conda-forge libglvnd-glx-cos7-x86_64 --force-reinstall
  conda install -y -c conda-forge xvfbwrapper --force-reinstall
  conda env config vars set LD_PRELOAD=/path/to/conda/envs/mujoco_env/x86_64-conda-linux-gnu/sysroot/usr/lib64/libGLdispatch.so.0
  ```
