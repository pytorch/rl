import torch
from torch.nn import Identity

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
            self.outdim = 512
            convnet = models.resnet18(pretrained=False)
        elif model_name == "resnet18":
            self.outdim = 512
            convnet = models.resnet34(pretrained=False)
        elif model_name == "resnet50":
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

    def _apply_transform(self, obs: torch.Tensor) -> None:
        return self.convnet(obs)

    @staticmethod
    def _load_weights(convnet, model_name):
        raise NotImplementedError


class R3MTransform(Compose):
    """
    TODO
    """

    def __init__(self, model_name, keys_in=None, keys_out=None, size=244):
        # ToTensor
        totensor = ToTensorImage(unsqueeze=False, keys_in=keys_in, keys_out=keys_out)
        keys_out = totensor.keys_out
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = ObservationNorm(
            keys_in=totensor.keys_out,
            loc=torch.tensor(mean).view(3, 1, 1),
            scale=torch.tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size)
        # R3M
        network = _R3MNet(in_keys=keys_out, out_keys=keys_out, model_name=model_name)
        transforms = [totensor, resize, normalize, network]
        super().__init__(*transforms)
