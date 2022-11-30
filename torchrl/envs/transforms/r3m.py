# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from tensordict import TensorDict
from torch.hub import load_state_dict_from_url
from torch.nn import Identity

from torchrl.data.tensor_specs import (
    CompositeSpec,
    NdUnboundedContinuousTensorSpec,
    TensorSpec,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import (
    CatTensors,
    Compose,
    FlattenObservation,
    ObservationNorm,
    Resize,
    ToTensorImage,
    Transform,
    UnsqueezeTransform,
)

try:
    from torchvision import models

    _has_tv = True
except ImportError:
    _has_tv = False


class _R3MNet(Transform):

    inplace = False

    def __init__(self, in_keys, out_keys, model_name, del_keys: bool = True):
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
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.convnet = convnet
        self.del_keys = del_keys

    def _call(self, tensordict):
        tensordict_view = tensordict.view(-1)
        super()._call(tensordict_view)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict

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

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError("_R3MNet can only infer CompositeSpec")

        keys = [key for key in observation_spec._specs.keys() if key in self.in_keys]
        device = observation_spec[keys[0]].device
        dim = observation_spec[keys[0]].shape[:-3]

        observation_spec = CompositeSpec(**observation_spec)
        if self.del_keys:
            for in_key in keys:
                del observation_spec[in_key]

        for out_key in self.out_keys:
            observation_spec[out_key] = NdUnboundedContinuousTensorSpec(
                shape=torch.Size([*dim, self.outdim]), device=device
            )

        return observation_spec

    @staticmethod
    def _load_weights(model_name, r3m_instance, dir_prefix):
        if model_name not in ("r3m_50", "r3m_34", "r3m_18"):
            raise ValueError(
                "model_name should be one of 'r3m_50', 'r3m_34' or 'r3m_18'"
            )
        # url = "https://download.pytorch.org/models/rl/r3m/" + model_name
        url = "https://pytorch.s3.amazonaws.com/models/rl/r3m/" + model_name + ".pt"
        d = load_state_dict_from_url(
            url,
            progress=True,
            map_location=next(r3m_instance.parameters()).device,
            model_dir=dir_prefix,
        )
        td = TensorDict(d["r3m"], []).unflatten_keys(".")
        td_flatten = td["module"]["convnet"].flatten_keys(".")
        state_dict = td_flatten.to_dict()
        r3m_instance.convnet.load_state_dict(state_dict)

    def load_weights(self, dir_prefix=None):
        self._load_weights(self.model_name, self, dir_prefix)


def _init_first(fun):
    def new_fun(self, *args, **kwargs):
        if not self.initialized:
            self._init()
        return fun(self, *args, **kwargs)

    return new_fun


class R3MTransform(Compose):
    """R3M Transform class.

    R3M provides pre-trained ResNet weights aimed at facilitating visual
    embedding for robotic tasks. The models are trained using Ego4d.

    See the paper:
        R3M: A Universal Visual Representation for Robot Manipulation (Suraj Nair,
            Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta)
            https://arxiv.org/abs/2203.12601

    The R3MTransform is created in a lazy manner: the object will be initialized
    only when an attribute (a spec or the forward method) will be queried.
    The reason for this is that the :obj:`_init()` method requires some attributes of
    the parent environment (if any) to be accessed: by making the class lazy we
    can ensure that the following code snippet works as expected:

    Examples:
        >>> transform = R3MTransform("resenet50", in_keys=["pixels"])
        >>> env.append_transform(transform)
        >>> # the forward method will first call _init which will look at env.observation_spec
        >>> env.reset()

    Args:
        model_name (str): one of resnet50, resnet34 or resnet18
        in_keys (list of str, optional): list of input keys. If left empty, the
            "pixels" key is assumed.
        out_keys (list of str, optional): list of output keys. If left empty,
             "r3m_vec" is assumed.
        size (int, optional): Size of the image to feed to resnet.
            Defaults to 244.
        download (bool, optional): if True, the weights will be downloaded using
            the torch.hub download API (i.e. weights will be cached for future use).
            Defaults to False.
        download_path (str, optional): path where to download the models.
            Default is None (cache path determined by torch.hub utils).
        tensor_pixels_keys (list of str, optional): Optionally, one can keep the
            original images (as collected from the env) in the output tensordict.
            If no value is provided, this won't be collected.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._is_3d = None
        cls.initialized = False
        cls._device = None
        cls._dtype = None
        return super().__new__(cls)

    def __init__(
        self,
        model_name: str,
        in_keys: List[str] = None,
        out_keys: List[str] = None,
        size: int = 244,
        stack_images: bool = True,
        download: bool = False,
        download_path: Optional[str] = None,
        tensor_pixels_keys: List[str] = None,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.download = download
        self.download_path = download_path
        self.model_name = model_name
        self.out_keys = out_keys
        self.size = size
        self.stack_images = stack_images
        self.tensor_pixels_keys = tensor_pixels_keys

    def _init(self):
        in_keys = self.in_keys
        model_name = self.model_name
        out_keys = self.out_keys
        size = self.size
        stack_images = self.stack_images
        tensor_pixels_keys = self.tensor_pixels_keys

        # ToTensor
        transforms = []
        if tensor_pixels_keys:
            for i in range(len(in_keys)):
                transforms.append(
                    CatTensors(
                        in_keys=[in_keys[i]],
                        out_key=tensor_pixels_keys[i],
                        del_keys=False,
                    )
                )

        totensor = ToTensorImage(
            unsqueeze=False,
            in_keys=in_keys,
        )
        transforms.append(totensor)

        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = ObservationNorm(
            in_keys=in_keys,
            loc=torch.tensor(mean).view(3, 1, 1),
            scale=torch.tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        transforms.append(normalize)

        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size, in_keys=in_keys)
        transforms.append(resize)

        # R3M
        if out_keys is None:
            if stack_images:
                out_keys = ["r3m_vec"]
            else:
                out_keys = [f"r3m_vec_{i}" for i in range(len(in_keys))]
        elif stack_images and len(out_keys) != 1:
            raise ValueError(
                f"out_key must be of length 1 if stack_images is True. Got out_keys={out_keys}"
            )
        elif not stack_images and len(out_keys) != len(in_keys):
            raise ValueError(
                "out_key must be of length equal to in_keys if stack_images is False."
            )

        if stack_images and len(in_keys) > 1:
            if self.is_3d:
                unsqueeze = UnsqueezeTransform(
                    in_keys=in_keys,
                    out_keys=in_keys,
                    unsqueeze_dim=-4,
                )
                transforms.append(unsqueeze)

            cattensors = CatTensors(
                in_keys,
                out_keys[0],
                dim=-4,
            )
            network = _R3MNet(
                in_keys=out_keys,
                out_keys=out_keys,
                model_name=model_name,
                del_keys=False,
            )
            flatten = FlattenObservation(-2, -1, out_keys)
            transforms = [*transforms, cattensors, network, flatten]
        else:
            network = _R3MNet(
                in_keys=in_keys,
                out_keys=out_keys,
                model_name=model_name,
                del_keys=True,
            )
            transforms = [*transforms, network]

        for transform in transforms:
            self.append(transform)
        if self.download:
            self[-1].load_weights(dir_prefix=self.download_path)
        self.initialized = True

        if self._device is not None:
            self.to(self._device)
        if self._dtype is not None:
            self.to(self._dtype)

    @property
    def is_3d(self):
        if self._is_3d is None:
            parent = self.parent
            for key in parent.observation_spec.keys():
                self._is_3d = len(parent.observation_spec[key].shape) == 3
                break
        return self._is_3d

    def to(self, dest: Union[DEVICE_TYPING, torch.dtype]):
        if isinstance(dest, torch.dtype):
            self._dtype = dest
        else:
            self._device = dest
        return super().to(dest)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    forward = _init_first(Compose.forward)
    transform_observation_spec = _init_first(Compose.transform_observation_spec)
    transform_input_spec = _init_first(Compose.transform_input_spec)
    transform_reward_spec = _init_first(Compose.transform_reward_spec)
    reset = _init_first(Compose.reset)
    init = _init_first(Compose.init)
