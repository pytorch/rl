from typing import Optional, Sequence

import torch

from torchrl.data.transforms import Transform, ObservationTransform

__all__ = ["VideoRecorder", "TensorDictRecorder"]


class VideoRecorder(ObservationTransform):
    """
    Video Recorder transform.
    Will record a series of observations from an environment and write them to a TensorBoard SummaryWriter object when needed.

    Args:
        writer (SummaryWriter): a tb.SummaryWriter instance where the video should be written.
        tag (str): the video tag in the writer.
        keys (Sequence[str], optional): keys to be read to produce the video.
            default: next_observation_pixels.
        skip (int): frame interval in the output video.
            default: 2
    """

    def __init__(
        self,
        writer: "SummaryWriter",
        tag: str,
        keys: Optional[Sequence[str]] = None,
        skip: int = 2,
        **kwargs,
    ) -> None:
        if keys is None:
            keys = ["next_observation_pixels"]

        super().__init__(keys=keys)
        video_kwargs = {"fps": 6}
        video_kwargs.update(kwargs)
        self.video_kwargs = video_kwargs
        self.iter = 0
        self.skip = skip
        self.writer = writer
        self.tag = tag
        self.count = 0
        self.obs = []
        try:
            import moviepy
        except:
            raise Exception(
                "moviepy not found, VideoRecorder cannot be created"
            )

    def _apply(self, observation: torch.Tensor) -> torch.Tensor:
        if not (observation.shape[-1] == 3 or observation.ndimension() == 2):
            raise RuntimeError(
                f"Invalid observation shape, got: {observation.shape}"
            )
        observation_trsf = observation
        self.count += 1
        if self.count % self.skip == 0:
            if observation.ndimension() == 2:
                observation_trsf = observation.unsqueeze(-3)
            else:
                if observation.ndimension() != 3:
                    raise RuntimeError(
                        "observation is expected to have 3 dimensions, "
                        f"got {observation.ndimension()} instead"
                    )
                if observation_trsf.shape[-1] != 3:
                    raise RuntimeError(
                        "observation_trsf is expected to have 3 dimensions, "
                        f"got {observation_trsf.ndimension()} instead"
                    )
                observation_trsf = observation_trsf.permute(2, 0, 1)
            self.obs.append(observation_trsf.cpu().to(torch.uint8))
        return observation

    def dump(self) -> None:
        """
        Writes the video to the self.writer attribute.

        Returns: None

        """
        self.writer.add_video(
            tag=f"{self.tag}",
            vid_tensor=torch.stack(self.obs, 0).unsqueeze(0),
            global_step=self.iter,
            **self.video_kwargs,
        )
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
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        if keys is None:
            keys = []

        super().__init__(keys=keys)
        self.iter = 0
        self.out_file_base = out_file_base
        self.td = []
        self.skip_reset = skip_reset
        self.skip = skip
        self.count = 0

    def _call(self, td: _TensorDict) -> _TensorDict:
        self.count += 1
        if self.count % self.skip == 0:
            _td = td
            if self.keys:
                _td = td.select(*self.keys).clone()
            self.td.append(_td)
        return td

    def dump(self) -> None:
        td = self.td
        if self.skip_reset:
            td = td[1:]
        torch.save(
            torch.stack(td, 0).contiguous(),
            f"{self.out_file_base}_tensor_dict.t",
        )
        self.iter += 1
        self.count = 0
        del self.td
        self.td = []
