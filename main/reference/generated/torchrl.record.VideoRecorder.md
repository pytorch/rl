# VideoRecorder

torchrl.record.VideoRecorder(*logger: Logger | [Service](torchrl.services.Service.html#torchrl.services.Service) | None*, *tag: str | None*, *in_keys: Sequence[NestedKey] | None = None*, *skip: int | None = None*, *center_crop: int | None = None*, *make_grid: bool | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *fps: int | None = None*, ***kwargs*) → None[[source]](../../_modules/torchrl/record/recorder.html#VideoRecorder)

Video Recorder transform.

Will record a series of observations from an environment and write them
to a Logger object when needed.

Parameters:

- **logger** (*Logger**or*[*Service*](torchrl.services.Service.html#torchrl.services.Service)) - a logger or logger-service owner where the video
should be written. To save the video under a memmap tensor or an mp4 file, use
the `CSVLogger` class.
- **tag** (*str*) - the video tag in the logger.
- **in_keys** (*Sequence**of**NestedKey**,**optional*) - keys to be read to produce the video.
Default is `"pixels"`.
- **skip** (*int*) - frame interval in the output video.
Defaults to `1` for vector environments and standalone use, and
`2` for a single parent environment.
- **center_crop** (*int**,**optional*) - value of square center crop.
- **make_grid** (*bool**,**optional*) - if `True`, a grid is created assuming that a
tensor of shape [B x W x H x 3] is provided, with B being the batch
size. Default is `True` if the transform has a parent environment, and `False`
if not.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - destination keys. Defaults
to `in_keys` if not provided.
- **fps** (*int**,**optional*) - Frames per second of the output video. Defaults to the logger predefined `fps`,
and overrides it if provided.
- ****kwargs** (*Dict**[**str**,**Any**]**,**optional*) - additional keyword arguments for
`log_video()`.

Examples

The following example shows how to save a rollout under a video. First a few imports:

```
>>> from torchrl.record import VideoRecorder
>>> from torchrl.record.loggers.csv import CSVLogger
>>> from torchrl.envs import TransformedEnv, DMControlEnv
```

The video format is chosen in the logger. Wandb and tensorboard will take care of that
on their own, CSV accepts various video formats.

```
>>> logger = CSVLogger(exp_name="cheetah", log_dir="cheetah_videos", video_format="mp4")
```

Some envs (eg, Atari games) natively return images, some require the user to ask for them.
Check [`GymEnv`](torchrl.envs.GymEnv.html#torchrl.envs.GymEnv) or [`DMControlEnv`](torchrl.envs.DMControlEnv.html#torchrl.envs.DMControlEnv) to see how to render images
in these contexts.

```
>>> base_env = DMControlEnv("cheetah", "run", from_pixels=True)
>>> env = TransformedEnv(base_env, VideoRecorder(logger=logger, tag="run_video"))
>>> env.rollout(100)
```

All transforms have a dump function, mostly a no-op except for `VideoRecorder`, and [`Compose`](torchrl.envs.transforms.Compose.html#torchrl.envs.transforms.Compose)
which will dispatch the dumps to all its members.

```
>>> env.transform.dump()
```

The transform can also be used within a dataset to save the video collected. Unlike in the environment case,
images will come in a batch. The `skip` argument will enable to save the images only at specific intervals.

```
>>> from torchrl.data.datasets import OpenXExperienceReplay
>>> from torchrl.envs import Compose
>>> from torchrl.record import VideoRecorder, CSVLogger
>>> # Create a logger that saves videos as mp4 using 24 frames per sec
>>> logger = CSVLogger("./dump", video_format="mp4", video_fps=24)
>>> # We use the VideoRecorder transform to save register the images coming from the batch.
>>> # Setting the fps to 12 overrides the one set in the logger, not doing so keeps it unchanged.
>>> t = VideoRecorder(logger=logger, tag="pixels", in_keys=[("next", "observation", "image")], fps=12)
>>> # Each batch of data will have 10 consecutive videos of 200 frames each (maximum, since strict_length=False)
>>> dataset = OpenXExperienceReplay("cmu_stretch", batch_size=2000, slice_len=200,
... download=True, strict_length=False,
... transform=t)
>>> # Get a batch of data and visualize it
>>> for data in dataset:
... t.dump()
... break
```

Our video is available under `./cheetah_videos/cheetah/videos/run_video_0.mp4`!