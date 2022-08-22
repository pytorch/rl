## Common Issues when rendering Mujoco Environments

1. RuntimeError with error stack like this when running jobs using schedulers like slurm:

```
    File "mjrendercontext.pyx", line 46, in mujoco_py.cymj.MjRenderContext.__init__

    File "mjrendercontext.pyx", line 114, in mujoco_py.cymj.    MjRenderContext._setup_opengl_context

    File "opengl_context.pyx", line 130, in mujoco_py.cymj.OffscreenOpenGLContext.__init__

RuntimeError: Failed to initialize OpenGL
```

> Mujoco's EGL code indexes devices globally while CUDA_VISIBLE_DEVICES (when used with job schedulers like slurm) returns the local device ids. This can be worked around by setting the `GPUS` environment variable to the global device id. For slurm, it can be obtained using `SLURM_STEP_GPUS` enviroment variable.

2. Rendered images are completely black.

> Make sure to call `env.render()` before reading the pixels.

3. `patchelf` dependency is missing.

> Install using `conda install patchelf`

4. Errors like "Onscreen rendering needs 101 device"

> Make sure to set `DISPLAY` environment variable correctly.
