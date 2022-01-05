import torch

from torchrl.data import TensorDict
from torchrl.data.transforms import Transform, ObservationTransform

__all__ = ["VideoRecorder", "TensorDictRecorder"]


class VideoRecorder(ObservationTransform):
    def __init__(self, writer, tag, keys=None, skip=2, **kwargs):
        if keys is None:
            keys = ["next_observation_pixels"]

        super().__init__(keys=keys)
        video_kwargs = {'fps': 6}
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
            raise Exception("moviepy not found, VideoRecorder cannot be created")

    def _apply(self, observation):
        if not (observation.shape[-1] == 3 or observation.ndimension() == 2):
            raise RuntimeError(f"Invalid observation shape, got: {observation.shape}")
        observation_trsf = observation
        self.count += 1
        if self.count % self.skip == 0:
            if observation.ndimension() == 2:
                observation_trsf = observation.unsqueeze(-3)
            else:
                if observation.ndimension() != 3:
                    raise RuntimeError("observation is expected to have 3 dimensions, "
                                       f"got {observation.ndimension()} instead")
                if observation_trsf.shape[-1] != 3:
                    raise RuntimeError("observation_trsf is expected to have 3 dimensions, "
                                       f"got {observation_trsf.ndimension()} instead")
                observation_trsf = observation_trsf.permute(2, 0, 1)
            self.obs.append(observation_trsf.cpu().to(torch.uint8))
        return observation

    def dump(self):
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
    def __init__(self, out_file_base, skip_reset=True, skip=4, keys=None):
        if keys is None:
            keys = []

        super().__init__(keys=keys)
        self.iter = 0
        self.out_file_base = out_file_base
        self.td = []
        self.skip_reset = skip_reset
        self.skip = skip
        self.count = 0

    def _call(self, td: TensorDict):
        self.count += 1
        if self.count % self.skip == 0:
            _td = td
            if self.keys:
                _td = td.select(*self.keys).clone()
            self.td.append(_td)
        return td

    def dump(self):
        td = self.td
        if self.skip_reset:
            td = td[1:]
        torch.save(torch.stack(td, 0), f"{self.out_file_base}_tensor_dict.t")
        self.iter += 1
        self.count = 0
        del self.td
        self.td = []
