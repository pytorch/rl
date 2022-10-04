# Working with [`habitat-lab`](https://github.com/facebookresearch/habitat-lab)

## Setting up your environment for habitat and torchrl

### Installing habitat

Instructions can be found on [habitat github repo](https://github.com/facebookresearch/habitat-lab).

  1. **Preparing conda env**

     Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
     ```bash
     # We require python>=3.7 and cmake>=3.10
     conda create -n habitat python=3.7 cmake=3.14.0
     conda activate habitat
     ```

  2. **conda install habitat-sim**
     - To install habitat-sim with bullet physics
        ```
        conda install habitat-sim withbullet -c conda-forge -c aihabitat
        ```
        See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

  3. **pip install habitat-lab stable version**.

        ```bash
        git clone https://github.com/facebookresearch/habitat-lab.git
        cd habitat-lab
        pip install -e habitat-lab  # install habitat_lab
        ```
  4. **Install habitat-baselines**.

      The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

        ```bash
        pip install -e habitat-baselines  # install habitat_baselines
        ```
### Installing TorchRL

Follow the instructions on the [README.md](../README.md).

## Common issues


1. `ImportError: /usr/lib/x86_64-linux-gnu/libOpenGL.so.0: undefined symbol: _glapi_tls_Current`
  **Solution**: as in [MUJOCO]([url](https://github.com/pytorch/rl/blob/main/knowledge_base/MUJOCO_INSTALLATION.md)) debug, Link conda to the right libOpenGL.so file (replace /path/to/conda and mujoco_env with the proper paths and names):
  ```shell
  conda install -y -c conda-forge libglvnd-glx-cos7-x86_64 --force-reinstall
  conda install -y -c conda-forge xvfbwrapper --force-reinstall
  conda env config vars set LD_PRELOAD=/path/to/conda/envs/mujoco_env/x86_64-conda-linux-gnu/sysroot/usr/lib64/libGLdispatch.so.0
  ```
