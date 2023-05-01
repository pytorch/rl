# Working with MuJoCo-based environments

From its [official repository](https://github.com/deepmind/mujoco/),
> MuJoCo stands for Multi-Joint dynamics with Contact. It is a general purpose 
> physics engine that aims to facilitate research and development in robotics, 
> biomechanics, graphics and animation, machine learning, and other areas which 
> demand fast and accurate simulation of articulated structures interacting with 
> their environment.

Recently, MuJoCo was [acquired and open-sourced](https://www.deepmind.com/blog/open-sourcing-mujoco) by DeepMind.
Since then, the library is accessible to whomever without licence. 
Python bindings were incorporated in the library, making the reliance on mujoco-py obsolete.
However, a series of libraries keep a legacy on the old mujoco bindings.

In this document, we detail issues and pro-tips for the new and old bindings of
the library.

## Installing MuJoCo

### Prerequisite for rendering (all mujoco versions)
MuJoCo offers some great rendering capabilities.
To do so, MuJoCo will use one of the following backends: glfw, osmesa or egl.
Of these, glfw will not work in headless environments. On the other hand, osmesa 
will not run on GPU. Therefore, our advice is to use the egl backend.

If you have a sudo access on your machine, you can install the following dependencies
to enable fast rendering:
```shell
$ sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
```
If you don't, these libraries can be installed via conda but be aware of the fact
that this is not the intended workflow and things may not work as expected:
```shell
$ conda activate mujoco_env
$ conda install -c conda-forge glew
$ conda install -c conda-forge mesalib
$ conda install -c anaconda mesa-libgl-cos6-x86_64
$ conda install -c menpo glfw3
```

In both cases, when running your code, you will want to tell mujoco which backend to use.
This can be done by setting the appropriate environment variables.
```shell
$ conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$ conda deactivate && conda activate mujoco_env
```

### New bindindgs (≥ 2.1.2)
You can install the pre-built binaries from the [mujoco release page](https://github.com/deepmind/mujoco/releases).
However, in most cases, you will only need the python bindings.
These can be installed via pip.
```shell
$ conda create -n mujoco_env python=3.9
$ conda activate mujoco_env
$ pip install mujoco
```

### Old bindings (≤ 2.1.1): mujoco-py
In some cases, you may need to use the old mujoco bindings. For instance, this 
be the case when using some legacy code that used mujoco-py instead of the new 
bindings, because of cluster requirements etc.
Refer to the [mujoco-py README.md](https://github.com/openai/mujoco-py#install-mujoco).
Using `conda`, your setup should look like this:
```shell
$ conda create -n mujoco_env python=3.9
$ conda activate mujoco_env
$ mkdir ~/.mujoco
$ cd ~/.mujoco
$ # check here for 2.1.0 versions https://github.com/deepmind/mujoco/releases/tag/2.1.0
$ # check here for earlier versions http://roboti.us/download.html
$ wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
$ tar -xf mujoco210-linux-x86_64.tar.gz
$ # for versions < 2.1.0, we need a licence file. Since mujoco is now free of 
$ # of charge, this can obtained easily
$ wget http://roboti.us/file/mjkey.txt
$ # let's tell conda about our mujoco repo
$ conda env config vars set MJLIB_PATH=/path/to/home/.mujoco/mujoco210/bin/libmujoco210.so \
$ > LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/home/.mujoco/mujoco210/bin \
$ > MUJOCO_PY_MUJOCO_PATH=/path/to/home/.mujoco/mujoco210
$ # For versions < 2.1.0, we must link the key too
$ conda env config vars set MUJOCO_PY_MJKEY_PATH=/path/to/home/.mujoco/mjkey.txt
$ # reload the env
$ conda deactivate && conda activate mujoco_env
```

#### Option 1: installing mujoco-py with pip
We **do not** recommend this as it may be hard to change code later on, since 
there are known issues when trying to use GPUs for rendering with native mujoco-py
code. Refer to Option 2 here below if that is the intended usage.
```shell
$ conda activate mujoco_env
$ pip install mujoco-py
```

#### Option 2: installing mujoco-py from a cloned repo
We recommend installing mujoco-py via cloning the repo and installing it 
locally. In case one must force mujoco-py to install against cuda or modify the 
path to the nvidia driver (especially with older versions of mujoco-py), cloning
the repo will facilitate those hacks.

```shell
$ conda activate mujoco_env
$ cd path/to/where/mujoco-py/must/be/cloned
$ git clone https://github.com/openai/mujoco-py
$ cd mujoco-py
$ python setup.py develop
$ # the following line of code needs to be adatped, depending on where nvidia drivers are located
$ conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

`mujoco-py` will execute some building operations the first time it is imported 
in a python script. 
This means that compatibility issues may go unnoticed until  you actually run 
your script for the first time.
To complete the installation, run the following commands:
```shell
$ python
>>> import mujoco_py
```
This should trigger the building pipeline.


**Sanity check**

To check that your mujoco-py has been built against the GPU, run
```python
>>> import mujoco_py
>>> print(mujoco_py.cymj) # check it has the tag: linuxgpuextensionbuilder
```
The result should contain a filename with the tag `linuxgpuextensionbuilder`.

## Common Issues during import or when rendering Mujoco Environments

The above setup will most likely cause some problems. We give a list of known 
issues when running `import mujoco_py` and some troubleshooting for each of them:

1. GL/glew.h not found
    ```
    /path/to/mujoco-py/mujoco_py/gl/eglshim.c:4:10: fatal error: GL/glew.h: No such file or directory
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
   ```

   _Solution_: install glew and glew-devel

   - Ubuntu: `sudo apt-get install libglew-dev libglew`
   - CentOS: `sudo yum install glew glew-devel`
   - Conda: `conda install -c conda-forge glew`

2. 
    ```
    include/GL/glu.h:38:10: fatal error: GL/gl.h: No such file or directory
      #include <GL/gl.h>
               ^~~~~~~~~
    ```

    _Solution_: This should disappear once `mesalib` is installed: `conda install -y -c conda-forge mesalib`
3. 
   ```
   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /path/to/conda/envs/compile/bin/../lib/libOSMesa.so.8)
   ```
   
   _Solution_: Install libgcc, e.g.: `conda install libgcc -y`. Then make sure that it is being loaded during execution:
   ```
   export LD_PRELOAD=$LD_PRELOAD:/path/to/conda/envs/compile/lib/libstdc++.so.6
   ```
4. 
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'
   ```

    _Solution_: `pip install patchelf`
5. 
    ```
    ImportError: /usr/lib/x86_64-linux-gnu/libOpenGL.so.0: undefined symbol: _glapi_tls_Current
    ```

    _Solution_: Link conda to the right `libOpenGL.so` file (replace `/path/to/conda` and `mujoco_env` with the proper paths and names):
    ```shelf
    conda install -y -c conda-forge libglvnd-glx-cos7-x86_64 --force-reinstall
    conda install -y -c conda-forge xvfbwrapper --force-reinstall
    conda env config vars set LD_PRELOAD=/path/to/conda/envs/mujoco_env/x86_64-conda-linux-gnu/sysroot/usr/lib64/libGLdispatch.so.0
    ```

6. 
    ```
    mujoco.FatalError: gladLoadGL error
    
    /path/to/conda/envs/mj_envs/lib/python3.8/site-packages/glfw/__init__.py:912: GLFWError: (65537) b'The GLFW library is not initialized'
    ```

    _Solution_: This can usually be sovled by setting EGL as your mujoco_gl backend: `MUJOCO_GL=egl python myscript.py`



7. RuntimeError with error stack like this when running jobs using schedulers like slurm:

```
    File "mjrendercontext.pyx", line 46, in mujoco_py.cymj.MjRenderContext.__init__

    File "mjrendercontext.pyx", line 114, in mujoco_py.cymj.    MjRenderContext._setup_opengl_context

    File "opengl_context.pyx", line 130, in mujoco_py.cymj.OffscreenOpenGLContext.__init__

RuntimeError: Failed to initialize OpenGL
```

> Mujoco's EGL code indexes devices globally while CUDA_VISIBLE_DEVICES 
  (when used with job schedulers like slurm) returns the local device ids. 
  This can be worked around by setting the `GPUS` environment variable to the 
  global device id. For slurm, it can be obtained using `SLURM_STEP_GPUS` enviroment variable.

8. Rendered images are completely black.

   _Solution_: Make sure to call `env.render()` before reading the pixels.

9. `patchelf` dependency is missing.

   _Solution_: Install using `conda install patchelf` or `pip install patchelf`

10. Errors like "Onscreen rendering needs 101 device"

    _Solution_: Make sure to set `DISPLAY` environment variable correctly.

11. `ImportError: Cannot initialize a headless EGL display.`

    _Solution_: Make sure you have installed mujoco and all its dependencies (see instructions above).
    Make sure you have set the `MUJOCO_GL=egl`.
    Make sure you have a GPU accessible on your machine.

12. `cannot find -lGL: No such file or directory`

    _Solution_: call `conda install -c anaconda mesa-libgl-devel-cos6-x86_64`

13. ```
    RuntimeError: Failed to initialize OpenGL
    ```

    _Solution_: Install libEGL:

    - Ubuntu: `sudo apt install libegl-dev libegl`
    - CentOS: `sudo yum install mesa-libEGL mesa-libEGL-devel`
    - Conda: `conda install -c anaconda mesa-libegl-cos6-x86_64`

14. ```
    fatal error: X11/Xlib.h: No such file or directory
       | #include <X11/Xlib.h>
       |          ^~~~~~~~~~~~
    ```

    _Solution_: Install X11:

    - Ubuntu: `sudo apt install libx11-dev`
    - CentOS: `sudo yum install libX11`
    - Conda: `conda install -c conda-forge xorg-libx11`

15. ```
    fatal error: GL/osmesa.h: No such file or directory
        1 | #include <GL/osmesa.h>
          |          ^~~~~~~~~~~~~
    compilation terminated.
    ```

    _Solution_: Install Osmesa:

    - Ubuntu: `sudo apt-get install libosmesa6-dev`
    - CentOS: `sudo yum install mesa-libOSMesa-devel`
    - Conda: `conda install -c menpo osmesa`
