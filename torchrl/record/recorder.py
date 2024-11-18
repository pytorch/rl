# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import math
from copy import copy
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from tensordict import NonTensorData, TensorDictBase

from tensordict.utils import NestedKey

from torchrl._utils import _can_be_pickled
from torchrl.data import TensorSpec
from torchrl.data.tensor_specs import NonTensor, Unbounded
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import EnvBase
from torchrl.envs.transforms import ObservationTransform, Transform
from torchrl.record.loggers import Logger

_has_tv = importlib.util.find_spec("torchvision", None) is not None


class VideoRecorder(ObservationTransform):
    """Video Recorder transform.

    Will record a series of observations from an environment and write them
    to a Logger object when needed.

    Args:
        logger (Logger): a Logger instance where the video
            should be written. To save the video under a memmap tensor or an mp4 file, use
            the :class:`~torchrl.record.loggers.CSVLogger` class.
        tag (str): the video tag in the logger.
        in_keys (Sequence of NestedKey, optional): keys to be read to produce the video.
            Default is :obj:`"pixels"`.
        skip (int): frame interval in the output video.
            Default is ``2`` if the transform has a parent environment, and ``1`` if not.
        center_crop (int, optional): value of square center crop.
        make_grid (bool, optional): if ``True``, a grid is created assuming that a
            tensor of shape [B x W x H x 3] is provided, with B being the batch
            size. Default is ``True`` if the transform has a parent environment, and ``False``
            if not.
        out_keys (sequence of NestedKey, optional): destination keys. Defaults
            to ``in_keys`` if not provided.

    Examples:
        The following example shows how to save a rollout under a video. First a few imports:

        >>> from torchrl.record import VideoRecorder
        >>> from torchrl.record.loggers.csv import CSVLogger
        >>> from torchrl.envs import TransformedEnv, DMControlEnv

        The video format is chosen in the logger. Wandb and tensorboard will take care of that
        on their own, CSV accepts various video formats.

        >>> logger = CSVLogger(exp_name="cheetah", log_dir="cheetah_videos", video_format="mp4")

        Some envs (eg, Atari games) natively return images, some require the user to ask for them.
        Check :class:`~torchrl.envs.GymEnv` or :class:`~torchrl.envs.DMControlEnv` to see how to render images
        in these contexts.

        >>> base_env = DMControlEnv("cheetah", "run", from_pixels=True)
        >>> env = TransformedEnv(base_env, VideoRecorder(logger=logger, tag="run_video"))
        >>> env.rollout(100)

        All transforms have a dump function, mostly a no-op except for ``VideoRecorder``, and :class:`~torchrl.envs.transforms.Compose`
        which will dispatch the `dumps` to all its members.

        >>> env.transform.dump()

        The transform can also be used within a dataset to save the video collected. Unlike in the environment case,
        images will come in a batch. The ``skip`` argument will enable to save the images only at specific intervals.

            >>> from torchrl.data.datasets import OpenXExperienceReplay
            >>> from torchrl.envs import Compose
            >>> from torchrl.record import VideoRecorder, CSVLogger
            >>> # Create a logger that saves videos as mp4
            >>> logger = CSVLogger("./dump", video_format="mp4")
            >>> # We use the VideoRecorder transform to save register the images coming from the batch.
            >>> t = VideoRecorder(logger=logger, tag="pixels", in_keys=[("next", "observation", "image")])
            >>> # Each batch of data will have 10 consecutive videos of 200 frames each (maximum, since strict_length=False)
            >>> dataset = OpenXExperienceReplay("cmu_stretch", batch_size=2000, slice_len=200,
            ...             download=True, strict_length=False,
            ...             transform=t)
            >>> # Get a batch of data and visualize it
            >>> for data in dataset:
            ...     t.dump()
            ...     break


    Our video is available under ``./cheetah_videos/cheetah/videos/run_video_0.mp4``!

    """

    def __init__(
        self,
        logger: Logger,
        tag: str,
        in_keys: Optional[Sequence[NestedKey]] = None,
        skip: int | None = None,
        center_crop: Optional[int] = None,
        make_grid: bool | None = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        **kwargs,
    ) -> None:
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        video_kwargs = {"fps": 6}
        video_kwargs.update(kwargs)
        self.video_kwargs = video_kwargs
        self.iter = 0
        self.skip = skip
        self.logger = logger
        self.tag = tag
        self.count = 0
        self.center_crop = center_crop
        self.make_grid = make_grid
        if center_crop and not _has_tv:
            raise ImportError(
                "Could not load center_crop from torchvision. Make sure torchvision is installed."
            )
        self.obs = []

    @property
    def make_grid(self):
        make_grid = self._make_grid
        if make_grid is None:
            if self.parent is not None:
                self._make_grid = True
                return True
            self._make_grid = False
            return False
        return make_grid

    @make_grid.setter
    def make_grid(self, value):
        self._make_grid = value

    @property
    def skip(self):
        skip = self._skip
        if skip is None:
            if self.parent is not None:
                self._skip = 2
                return 2
            self._skip = 1
            return 1
        return skip

    @skip.setter
    def skip(self, value):
        self._skip = value

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        if isinstance(observation, NonTensorData):
            observation_trsf = torch.tensor(observation.data)
        else:
            observation_trsf = observation
        self.count += 1
        if self.count % self.skip == 0:
            if (
                observation_trsf.ndim >= 3
                and observation_trsf.shape[-3] == 3
                and observation_trsf.shape[-2] > 3
                and observation_trsf.shape[-1] > 3
            ):
                # permute the channels to the last dim
                observation_trsf = observation_trsf.permute(
                    *range(observation_trsf.ndim - 3), -2, -1, -3
                )
            if not (
                observation_trsf.shape[-1] == 3 or observation_trsf.ndimension() == 2
            ):
                raise RuntimeError(
                    f"Invalid observation shape, got: {observation.shape}"
                )
            observation_trsf = observation_trsf.clone()

            if observation.ndimension() == 2:
                observation_trsf = observation.unsqueeze(-3)
            else:
                if observation_trsf.shape[-1] != 3:
                    raise RuntimeError(
                        "observation_trsf is expected to have 3 dimensions, "
                        f"got {observation_trsf.ndimension()} instead"
                    )
                trailing_dim = range(observation_trsf.ndimension() - 3)
                observation_trsf = observation_trsf.permute(*trailing_dim, -1, -3, -2)
            if self.center_crop:
                if not _has_tv:
                    raise ImportError(
                        "Could not import torchvision, `center_crop` not available. "
                        "Make sure torchvision is installed in your environment."
                    )
                from torchvision.transforms.functional import (
                    center_crop as center_crop_fn,
                )

                observation_trsf = center_crop_fn(
                    observation_trsf, [self.center_crop, self.center_crop]
                )
            if self.make_grid and observation_trsf.ndimension() >= 4:
                if not _has_tv:
                    raise ImportError(
                        "Could not import torchvision, `make_grid` not available. "
                        "Make sure torchvision is installed in your environment."
                    )
                from torchvision.utils import make_grid

                obs_flat = observation_trsf.flatten(0, -4)
                observation_trsf = make_grid(
                    obs_flat, nrow=int(math.ceil(math.sqrt(obs_flat.shape[0])))
                )
                self.obs.append(observation_trsf.to("cpu", torch.uint8))
            elif observation_trsf.ndimension() >= 4:
                self.obs.extend(observation_trsf.to("cpu", torch.uint8).flatten(0, -4))
            else:
                self.obs.append(observation_trsf.to("cpu", torch.uint8))
        return observation

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def dump(self, suffix: Optional[str] = None) -> None:
        """Writes the video to the ``self.logger`` attribute.

        Calling ``dump`` when no image has been stored in a no-op.

        Args:
            suffix (str, optional): a suffix for the video to be recorded
        """
        if self.obs:
            obs = torch.stack(self.obs, 0).unsqueeze(0).cpu()
        else:
            obs = None
        self.obs = []
        if obs is not None:
            if suffix is None:
                tag = self.tag
            else:
                tag = "_".join([self.tag, suffix])
            if self.logger is not None:
                self.logger.log_video(
                    name=tag,
                    video=obs,
                    step=self.iter,
                    **self.video_kwargs,
                )
        self.iter += 1
        self.count = 0
        self.obs = []

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self._call(tensordict_reset)
        return tensordict_reset


class TensorDictRecorder(Transform):
    """TensorDict recorder.

    When the 'dump' method is called, this class will save a stack of the tensordict resulting from :obj:`env.step(td)` in a
    file with a prefix defined by the out_file_base argument.

    Args:
        out_file_base (str): a string defining the prefix of the file where the tensordict will be written.
        skip_reset (bool): if ``True``, the first TensorDict of the list will be discarded (usually the tensordict
            resulting from the call to :obj:`env.reset()`)
            default: True
        skip (int): frame interval for the saved tensordict.
            default: 4

    """

    def __init__(
        self,
        out_file_base: str,
        skip_reset: bool = True,
        skip: int = 4,
        in_keys: Optional[Sequence[str]] = None,
    ) -> None:
        if in_keys is None:
            in_keys = []

        super().__init__(in_keys=in_keys)
        self.iter = 0
        self.out_file_base = out_file_base
        self.td = []
        self.skip_reset = skip_reset
        self.skip = skip
        self.count = 0

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.count += 1
        if self.count % self.skip == 0:
            _td = tensordict
            if self.in_keys:
                _td = tensordict.select(*self.in_keys).to_tensordict()
            self.td.append(_td)
        return tensordict

    def dump(self, suffix: Optional[str] = None) -> None:
        if suffix is None:
            tag = self.tag
        else:
            tag = "_".join([self.tag, suffix])

        td = self.td
        if self.skip_reset:
            td = td[1:]
        torch.save(
            torch.stack(td, 0).contiguous(),
            f"{tag}_tensordict.t",
        )
        self.iter += 1
        self.count = 0
        del self.td
        self.td = []

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self._call(tensordict_reset)
        return tensordict_reset


class PixelRenderTransform(Transform):
    """A transform to call render on the parent environment and register the pixel observation in the tensordict.

    This transform offers an alternative to the ``from_pixels`` syntatic sugar when instantiating an environment
    that offers rendering is expensive, or when ``from_pixels`` is not implemented.
    It can be used within a single environment or over batched environments alike.

    Args:
        out_keys (List[NestedKey] or Nested): List of keys where to register the pixel observations.
        preproc (Callable, optional): a preproc function. Can be used to reshape the observation, or apply
            any other transformation that makes it possible to register it in the output data.
        as_non_tensor (bool, optional): if ``True``, the data will be written as a :class:`~tensordict.NonTensorData`
            thereby relaxing the shape requirements. If not provided, it will be inferred automatically from the
            input data type and shape.
        render_method (str, optional): the name of the render method. Defaults to ``"render"``.
        pass_tensordict (bool, optional): if ``True``, the input tensordict will be passed to the
            render method. This enables rendering for stateless environments. Defaults to ``False``.
        **kwargs: additional keyword arguments to pass to the render function (e.g. ``mode="rgb_array"``).

    Examples:
        >>> from torchrl.envs import GymEnv, check_env_specs, ParallelEnv, EnvCreator
        >>> from torchrl.record.loggers import CSVLogger
        >>> from torchrl.record.recorder import PixelRenderTransform, VideoRecorder
        >>>
        >>> def make_env():
        >>>     env = GymEnv("CartPole-v1", render_mode="rgb_array")
        >>>     env = env.append_transform(PixelRenderTransform())
        >>>     return env
        >>>
        >>> if __name__ == "__main__":
        ...     logger = CSVLogger("dummy", video_format="mp4")
        ...
        ...     env = ParallelEnv(4, EnvCreator(make_env))
        ...
        ...     env = env.append_transform(VideoRecorder(logger=logger, tag="pixels_record"))
        ...     env.rollout(3)
        ...
        ...     check_env_specs(env)
        ...
        ...     r = env.rollout(30)
        ...     print(env)
        ...     env.transform.dump()
        ...     env.close()

    This transform can also be used whenever a batched environment ``render()`` returns a single image:

    Examples:
        >>> from torchrl.envs import check_env_specs
        >>> from torchrl.envs.libs.vmas import VmasEnv
        >>> from torchrl.record.loggers import CSVLogger
        >>> from torchrl.record.recorder import PixelRenderTransform, VideoRecorder
        >>>
        >>> env = VmasEnv(
        ...     scenario="flocking",
        ...     num_envs=32,
        ...     continuous_actions=True,
        ...     max_steps=200,
        ...     device="cpu",
        ...     seed=None,
        ...     # Scenario kwargs
        ...     n_agents=5,
        ... )
        >>>
        >>> logger = CSVLogger("dummy", video_format="mp4")
        >>>
        >>> env = env.append_transform(PixelRenderTransform(mode="rgb_array", preproc=lambda x: x.copy()))
        >>> env = env.append_transform(VideoRecorder(logger=logger, tag="pixels_record"))
        >>>
        >>> check_env_specs(env)
        >>>
        >>> r = env.rollout(30)
        >>> env.transform[-1].dump()

    The transform can be disabled using the :meth:`~torchrl.record.PixelRenderTransform.switch` method, which will
    turn the rendering on if it's off or off if it's on (an argument can also be passed to control this behavior).
    Since transforms are :class:`~torch.nn.Module` instances, :meth:`~torch.nn.Module.apply` can be used to control
    this behavior:

        >>> def switch(module):
        ...     if isinstance(module, PixelRenderTransform):
        ...         module.switch()
        >>> env.apply(switch)

    """

    def __init__(
        self,
        out_keys: List[NestedKey] = None,
        preproc: Callable[
            [np.ndarray | torch.Tensor], np.ndarray | torch.Tensor
        ] = None,
        as_non_tensor: bool = None,
        render_method: str = "render",
        pass_tensordict: bool = False,
        **kwargs,
    ) -> None:
        if out_keys is None:
            out_keys = ["pixels"]
        elif isinstance(out_keys, (str, tuple)):
            out_keys = [out_keys]
        if len(out_keys) != 1:
            raise RuntimeError(
                f"Expected one and only one out_key, got out_keys={out_keys}"
            )
        if preproc is not None and not _can_be_pickled(preproc):
            preproc = CloudpickleWrapper(preproc)
        self.preproc = preproc
        self.as_non_tensor = as_non_tensor
        self.kwargs = kwargs
        self.render_method = render_method
        self._enabled = True
        self.pass_tensordict = pass_tensordict
        super().__init__(in_keys=[], out_keys=out_keys)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._enabled:
            return tensordict

        method = getattr(self.parent, self.render_method)
        if not self.pass_tensordict:
            array = method(**self.kwargs)
        else:
            array = method(tensordict, **self.kwargs)

        if self.preproc:
            array = self.preproc(array)
        if self.as_non_tensor is None:
            if isinstance(array, list):
                if isinstance(array[0], np.ndarray):
                    array = np.asarray(array)
                else:
                    array = torch.as_tensor(array)
            if (
                array.ndim == 3
                and array.shape[-1] == 3
                and self.parent.batch_size != ()
            ):
                self.as_non_tensor = True
            else:
                self.as_non_tensor = False
        if not self.as_non_tensor:
            try:
                tensordict.set(self.out_keys[0], array)
            except Exception:
                raise RuntimeError(
                    f"An exception was raised while writing the rendered array "
                    f"(shape={getattr(array, 'shape', None)}, dtype={getattr(array, 'dtype', None)}) in the tensordict with shape {tensordict.shape}. "
                    f"Consider adapting your preproc function in {type(self).__name__}. You can also "
                    f"pass keyword arguments to the render function of the parent environment, or save "
                    f"this observation as a non-tensor data with as_non_tensor=True."
                )
        else:
            tensordict.set_non_tensor(self.out_keys[0], array)
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        # Adds the pixel observation spec by calling render on the parent env
        switch = False
        if not self.enabled:
            switch = True
            self.switch()
        parent = self.parent
        td_in = parent.reset()
        self._call(td_in)
        obs = td_in.get(self.out_keys[0])
        if isinstance(obs, NonTensorData):
            spec = NonTensor(device=obs.device, dtype=obs.dtype, shape=obs.shape)
        else:
            spec = Unbounded(device=obs.device, dtype=obs.dtype, shape=obs.shape)
        observation_spec[self.out_keys[0]] = spec
        if switch:
            self.switch()
        return observation_spec

    def switch(self, mode: str | bool = None):
        """Sets the transform on or off.

        Args:
            mode (str or bool, optional): if provided, sets the switch to the desired mode.
                ``"on"``, ``"off"``, ``True`` and ``False`` are accepted values.
                By default, ``switch`` sets the mode to the opposite of the current one.

        """
        if mode is None:
            mode = not self._enabled
        if not isinstance(mode, bool):
            if mode not in ("on", "off"):
                raise ValueError("mode must be either 'on' or 'off', or a boolean.")
            mode = mode == "on"
        self._enabled = mode

    @property
    def enabled(self) -> bool:
        """Whether the recorder is enabled."""
        return self._enabled

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        out = super().set_container(container)
        if isinstance(self.parent, EnvBase):
            # Start the env if needed
            method = getattr(self.parent, self.render_method, None)
            if method is None or not callable(method):
                raise ValueError(
                    f"The render method must exist and be a callable. Got render={method}."
                )
        return out
