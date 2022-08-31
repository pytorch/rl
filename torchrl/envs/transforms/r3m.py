from typing import List

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
        super().__init__(keys_in=in_keys, keys_out=out_keys)
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
    """R3M Transform class.

    R3M provides pre-trained ResNet weights aimed at facilitating visual
    embedding for robotic tasks. The models are trained using Ego4d.
    See the paper:
        R3M: A Universal Visual Representation for Robot Manipulation (Suraj Nair,
            Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta)
            https://arxiv.org/abs/2203.12601

    Args:
        model_name (str): one of resnet50, resnet34 or resnet18
        keys_in (list of str, optional): list of input keys. If left empty, the
            "next_pixels" key is assumed.
        keys_out (list of str, optional): list of output keys. If left empty,
             "next_r3m_vec" is assumed.
        size (int, optional): Size of the image to feed to resnet.
            Defaults to 244.
        download (bool, optional): if True, the weights will be downloaded using
            the torch.hub download API (i.e. weights will be cached for future use).
            Defaults to False.
        tensor_pixels_key (str, optional): Optionally, one can keep the intermediate
            image transform (after normalization) in the output tensordict.
            If no value is provided, this won't be collected.
    """

    def __init__(
        self,
        model_name: str,
        keys_in: List[str] = None,
        keys_out: List[str] = None,
        size: int = 244,
        download: bool = False,
        tensor_pixels_key: str = None,
    ):
        self.download = download
        # ToTensor
        if tensor_pixels_key is None:
            tensor_pixels_key = keys_in
        else:
            tensor_pixels_key = [tensor_pixels_key]
        totensor = ToTensorImage(
            unsqueeze=False, keys_in=keys_in, keys_out=tensor_pixels_key
        )
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = ObservationNorm(
            keys_in=tensor_pixels_key,
            loc=torch.tensor(mean).view(3, 1, 1),
            scale=torch.tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size)
        # R3M
        if keys_out is None:
            keys_out = ["next_r3m_vec"]
        network = _R3MNet(
            in_keys=tensor_pixels_key, out_keys=keys_out, model_name=model_name
        )
        transforms = [totensor, resize, normalize, network]
        super().__init__(*transforms)
        if self.download:
            self[-1].load_weights()
