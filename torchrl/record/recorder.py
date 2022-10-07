# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch

try:
    from torchvision.transforms.functional import center_crop as center_crop_fn
    from torchvision.utils import make_grid
except ImportError:
    center_crop_fn = None

from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs.transforms import ObservationTransform, Transform
from torchrl.trainers.loggers import Logger

__all__ = ["VideoRecorder", "TensorDictRecorder"]


class VideoRecorder(ObservationTransform):
    """
    Video Recorder transform.
    Will record a series of observations from an environment and write them
    to a Logger object when needed.

    Args:
        logger (Logger): a Logger instance where the video
            should be written.
        tag (str): the video tag in the logger.
        keys_in (Sequence[str], optional): keys to be read to produce the video.
            Default is `"next_pixels"`.
        skip (int): frame interval in the output video.
            Default is 2.
        center_crop (int, optional): value of square center crop.
        make_grid (bool, optional): if True, a grid is created assuming that a
            tensor of shape [B x W x H x 3] is provided, with B being the batch
            size. Default is True.
    """

    def __init__(
        self,
        logger: Logger,
        tag: str,
        keys_in: Optional[Sequence[str]] = None,
        skip: int = 2,
        center_crop: Optional[int] = None,
        make_grid: bool = True,
        **kwargs,
    ) -> None:
        if keys_in is None:
            keys_in = ["next_pixels"]

        super().__init__(keys_in=keys_in)
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
        if center_crop and not center_crop_fn:
            raise ImportError(
                "Could not load center_crop from torchvision. Make sure torchvision is installed."
            )
        self.obs = []

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        if not (observation.shape[-1] == 3 or observation.ndimension() == 2):
            raise RuntimeError(f"Invalid observation shape, got: {observation.shape}")
        observation_trsf = observation.clone()
        self.count += 1
        if self.count % self.skip == 0:
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
                if center_crop_fn is None:
                    raise ImportError(
                        "Could not import torchvision, `center_crop` not available."
                        "Make sure torchvision is installed in your environment."
                    )
                observation_trsf = center_crop_fn(
                    observation_trsf, [self.center_crop, self.center_crop]
                )
            if self.make_grid and observation_trsf.ndimension() == 4:
                if make_grid is None:
                    raise ImportError(
                        "Could not import torchvision, `make_grid` not available."
                        "Make sure torchvision is installed in your environment."
                    )
                observation_trsf = make_grid(observation_trsf)
            self.obs.append(observation_trsf.to(torch.uint8))
        return observation

    def dump(self, suffix: Optional[str] = None) -> None:
        """Writes the video to the self.logger attribute.

        Args:
            suffix (str, optional): a suffix for the video to be recorded
        """
        if suffix is None:
            tag = self.tag
        else:
            tag = "_".join([self.tag, suffix])
        obs = torch.stack(self.obs, 0).unsqueeze(0).cpu()
        del self.obs
        if self.logger is not None:
            self.logger.log_video(
                name=tag,
                video=obs,
                step=self.iter,
                **self.video_kwargs,
            )
        del obs
        self.iter += 1
        self.count = 0
        self.obs = []


class TensorDictRecorder(Transform):
    """
    TensorDict recorder.
    When the 'dump' method is called, this class will save a stack of the tensordict resulting from `env.step(td)` in a
    file with a prefix defined by the out_file_base argument.

    Args:
        out_file_base (str): a string defining the prefix of the file where the tensordict will be written.
        skip_reset (bool): if True, the first TensorDict of the list will be discarded (usually the tensordict
            resulting from the call to `env.reset()`)
            default: True
        skip (int): frame interval for the saved tensordict.
            default: 4

    """

    def __init__(
        self,
        out_file_base: str,
        skip_reset: bool = True,
        skip: int = 4,
        keys_in: Optional[Sequence[str]] = None,
    ) -> None:
        if keys_in is None:
            keys_in = []

        super().__init__(keys_in=keys_in)
        self.iter = 0
        self.out_file_base = out_file_base
        self.td = []
        self.skip_reset = skip_reset
        self.skip = skip
        self.count = 0

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        self.count += 1
        if self.count % self.skip == 0:
            _td = td
            if self.keys_in:
                _td = td.select(*self.keys_in).to_tensordict()
            self.td.append(_td)
        return td

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
