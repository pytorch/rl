.. currentmodule:: torchrl.record

.. _Environment-Recorders:

Recorders
=========

Recording data during environment rollout execution is crucial to keep an eye on the algorithm performance as well as
reporting results after training.

TorchRL offers several tools to interact with the environment output: first and foremost, a ``callback`` callable
can be passed to the :meth:`~torchrl.envs.EnvBase.rollout` method. This function will be called upon the collected
tensordict at each iteration of the rollout (if some iterations have to be skipped, an internal variable should be added
to keep track of the call count within ``callback``).

To save collected tensordicts on disk, the :class:`~torchrl.record.TensorDictRecorder` can be used.

Recording videos
----------------

Several backends offer the possibility of recording rendered images from the environment.
If the pixels are already part of the environment output (e.g. Atari or other game simulators), a
:class:`~torchrl.record.VideoRecorder` can be appended to the environment. This environment transform takes as input
a logger capable of recording videos (e.g. :class:`~torchrl.record.loggers.CSVLogger`, :class:`~torchrl.record.loggers.WandbLogger`
or :class:`~torchrl.record.loggers.TensorBoardLogger`) as well as a tag indicating where the video should be saved.
For instance, to save mp4 videos on disk, one can use :class:`~torchrl.record.loggers.CSVLogger` with a `video_format="mp4"`
argument.

The :class:`~torchrl.record.VideoRecorder` transform can handle batched images and automatically detects numpy or PyTorch
formatted images (WHC or CWH).

    >>> logger = CSVLogger("dummy-exp", video_format="mp4")
    >>> env = GymEnv("ALE/Pong-v5")
    >>> env = env.append_transform(VideoRecorder(logger, tag="rendered", in_keys=["pixels"]))
    >>> env.rollout(10)
    >>> env.transform.dump()  # Save the video and clear cache

Note that the cache of the transform will keep on growing until dump is called. It is the user responsibility to
take care of calling `dump` when needed to avoid OOM issues.

In some cases, creating a testing environment where images can be collected is tedious or expensive, or simply impossible
(some libraries only allow one environment instance per workspace).
In these cases, assuming that a `render` method is available in the environment, the :class:`~torchrl.record.PixelRenderTransform`
can be used to call `render` on the parent environment and save the images in the rollout data stream.
This class works over single and batched environments alike:

    >>> from torchrl.envs import GymEnv, check_env_specs, ParallelEnv, EnvCreator
    >>> from torchrl.record.loggers import CSVLogger
    >>> from torchrl.record.recorder import PixelRenderTransform, VideoRecorder
    >>>
    >>> def make_env():
    >>>     env = GymEnv("CartPole-v1", render_mode="rgb_array")
    >>>     # Uncomment this line to execute per-env
    >>>     # env = env.append_transform(PixelRenderTransform())
    >>>     return env
    >>>
    >>> if __name__ == "__main__":
    ...     logger = CSVLogger("dummy", video_format="mp4")
    ...
    ...     env = ParallelEnv(16, EnvCreator(make_env))
    ...     env.start()
    ...     # Comment this line to execute per-env
    ...     env = env.append_transform(PixelRenderTransform())
    ...
    ...     env = env.append_transform(VideoRecorder(logger=logger, tag="pixels_record"))
    ...     env.rollout(3)
    ...
    ...     check_env_specs(env)
    ...
    ...     r = env.rollout(30)
    ...     env.transform.dump()
    ...     env.close()


Recorders are transforms that register data as they come in, for logging purposes.

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    TensorDictRecorder
    VideoRecorder
    PixelRenderTransform
