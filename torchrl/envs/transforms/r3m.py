import torch
from torch.hub import load_state_dict_from_url
from torch.nn import Identity

from torchrl.data import TensorDict
from torchrl.envs.transforms import (
    ToTensorImage,
    Compose,
    ObservationNorm,
    Resize,
    Transform,
)

try:
    from torchvision import models

    _has_tv = True
except ImportError:
    _has_tv = False

__all__ = ["R3MTransform"]


class _R3MNet(Transform):

    inplace = False

    def __init__(self, in_keys, out_keys, model_name):
        if not _has_tv:
            raise ImportError(
                "Tried to instantiate R3M without torchvision. Make sure you have "
                "torchvision installed in your environment."
            )
        if model_name == "resnet18":
            self.model_name = "r3m_18"
            self.outdim = 512
            convnet = models.resnet18(pretrained=False)
        elif model_name == "resnet34":
            self.model_name = "r3m_34"
            self.outdim = 512
            convnet = models.resnet34(pretrained=False)
        elif model_name == "resnet50":
            self.model_name = "r3m_50"
            self.outdim = 2048
            convnet = models.resnet50(pretrained=False)
        else:
            raise NotImplementedError(
                f"model {model_name} is currently not supported by R3M"
            )
        convnet.fc = Identity()
        super().__init__(keys_in=in_keys)
        self.convnet = convnet

    def _call(self, tensordict):
        tensordict_view = tensordict.view(-1)
        return super()._call(tensordict_view)

    @torch.no_grad()
    def _apply_transform(self, obs: torch.Tensor) -> None:
        shape = None
        if obs.ndimension() > 4:
            shape = obs.shape[:-3]
            obs = obs.flatten(0, -4)
        out = self.convnet(obs)
        if shape is not None:
            out = out.view(*shape, *out.shape[1:])
        return out

    @staticmethod
    def _load_weights(model_name, r3m_instance):
        if model_name not in ("r3m_50", "r3m_34", "r3m_18"):
            raise ValueError(
                "model_name should be one of 'r3m_50', 'r3m_34' or 'r3m_18'"
            )
        # url = "https://download.pytorch.org/models/rl/r3m/" + model_name
        url = "https://pytorch.s3.amazonaws.com/models/rl/r3m/" + model_name + ".pt"
        d = load_state_dict_from_url(
            url, progress=True, map_location=next(r3m_instance.parameters()).device
        )
        td = TensorDict(d["r3m"], []).unflatten_keys(".")
        td_flatten = td["module"]["convnet"].flatten_keys(".")
        state_dict = td_flatten.to_dict()
        r3m_instance.convnet.load_state_dict(state_dict)

    def load_weights(self):
        self._load_weights(self.model_name, self)


class R3MTransform(Compose):
    """
    TODO
    """

    def __init__(
        self,
        model_name,
        keys_in=None,
        keys_out=None,
        size=244,
        download=False,
        tensor_pixel_key=None,
    ):
        self.download = download
        # ToTensor
        if tensor_pixel_key is None:
            tensor_pixel_key = keys_in
        totensor = ToTensorImage(
            unsqueeze=False, keys_in=keys_in, keys_out=tensor_pixel_key
        )
        keys_out = totensor.keys_out
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = ObservationNorm(
            keys_in=tensor_pixel_key,
            loc=torch.tensor(mean).view(3, 1, 1),
            scale=torch.tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size)
        # R3M
        network = _R3MNet(
            in_keys=tensor_pixel_key, out_keys=keys_out, model_name=model_name
        )
        transforms = [totensor, resize, normalize, network]
        super().__init__(*transforms)
        if self.download:
            self[-1].load_weights()
