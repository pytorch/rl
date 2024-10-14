# Customising Video Renders

## Tweaking Video Rendering Settings
TorchRL relies heavily on the [torchvision.io](https://pytorch.org/vision/main/io.html) 
and [PyAV](https://github.com/PyAV-Org/PyAV) modules for its video logging 
capabilities. Though these libraries are quite convenient and powerful, it is 
not easy to access the variety of knobs and settings at your disposal. 

This guide hopes to clarify what appear to be the general principles behind
customising video rendering, and show you how you can manually adjust your 
rollouts' rendering settings to your liking.

## General Principles
Ultimately, [torchvision.io](https://pytorch.org/vision/main/io.html) and 
[PyAV](https://github.com/PyAV-Org/PyAV) make calls to [FFmpeg](https://ffmpeg.org/)
libraries in order to render videos. 

In other words:

- Whatever can be fed into [FFmpeg](https://ffmpeg.org/), we can also feed 
into TorchRL's `Loggers`.
- For any custom settings we wish to use, we must reference them from 
[FFmpeg's documentation](https://trac.ffmpeg.org/)

## Video Rendering Customization Example

Suppose the following snippet gave us extremely blurry videos, even though 
we provided it clear, frame-by-frame images to stitch together:
```python
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.record import CSVLogger, VideoRecorder

logger = CSVLogger(exp_name="my_exp")
env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)

recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)
rollout = record_env.rollout(max_steps=3)
recorder.dump()
```

Since TorchRL's default video codec is [H264](https://trac.ffmpeg.org/wiki/Encode/H.264),
the settings that we must change should be in there.

For the purposes of this example, let us choose a 
[Constant Rate Factor (CRF)](https://trac.ffmpeg.org/wiki/Encode/H.264#crf) of 
`17` and a [preset](https://trac.ffmpeg.org/wiki/Encode/H.264#Preset) of `slow`,
as advised by the documentation.

We can improve the video quality by appending all our desired settings 
(as keyword arguments) to `recorder` like so:
```python
recorder = VideoRecorder(logger, tag = "my_video", options = {"crf": "17", "preset": "slow"})
```
