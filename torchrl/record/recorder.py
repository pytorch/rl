# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from copy import copy
from typing import Optional, Sequence

import torch

from tensordict import TensorDictBase

from tensordict.utils import NestedKey

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
        self.count += 1
        if self.count % self.skip == 0:
            if (
                observation.ndim >= 3
                and observation.shape[-3] == 3
                and observation.shape[-2] > 3
                and observation.shape[-1] > 3
            ):
                # permute the channels to the last dim
                observation_trsf = observation.permute(
                    *range(observation.ndim - 3), -2, -1, -3
                )
            else:
                observation_trsf = observation
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
                        "Could not import torchvision, `center_crop` not available."
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
                        "Could not import torchvision, `make_grid` not available."
                        "Make sure torchvision is installed in your environment."
                    )
                from torchvision.utils import make_grid

                observation_trsf = make_grid(observation_trsf.flatten(0, -4))
                self.obs.append(observation_trsf.to(torch.uint8))
            elif observation_trsf.ndimension() >= 4:
                self.obs.extend(observation_trsf.to(torch.uint8).flatten(0, -4))
            else:
                self.obs.append(observation_trsf.to(torch.uint8))
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
