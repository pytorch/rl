# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from torch.hub import load_state_dict_from_url

from torchrl.data import TensorDict, DEVICE_TYPING
from torchrl.data.tensor_specs import (
    TensorSpec,
    CompositeSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.envs.transforms import (
    ToTensorImage,
    Compose,
    ObservationNorm,
    Resize,
    Transform,
    CatTensors,
    FlattenObservation,
    UnsqueezeTransform,
)

try:
    from torchvision import models

    _has_tv = True
except ImportError:
    _has_tv = False


class _VIPNet(Transform):

    inplace = False

    def __init__(self, in_keys, out_keys, model_name="resnet50", del_keys: bool = True):
        if not _has_tv:
            raise ImportError(
                "Tried to instantiate VIP without torchvision. Make sure you have "
                "torchvision installed in your environment."
            )
        if model_name == "resnet50":
            self.model_name = "vip_50"
            self.outdim = 2048
            convnet = models.resnet50(pretrained=False)
            convnet.fc = torch.nn.Linear(self.outdim, 1024)
        else:
            raise NotImplementedError(
                f"model {model_name} is currently not supported by VIP"
            )
        super().__init__(keys_in=in_keys, keys_out=out_keys)
        self.convnet = convnet
        self.del_keys = del_keys

    def _call(self, tensordict):
        tensordict_view = tensordict.view(-1)
        super()._call(tensordict_view)
        if self.del_keys:
            tensordict.exclude(*self.keys_in, inplace=True)
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
            raise ValueError("_VIPNet can only infer CompositeSpec")

        keys = [key for key in observation_spec._specs.keys() if key in self.keys_in]
        device = observation_spec[keys[0]].device

        observation_spec = CompositeSpec(**observation_spec)
        if self.del_keys:
            for key_in in keys:
                del observation_spec[key_in]

        for key_out in self.keys_out:
            observation_spec[key_out] = NdUnboundedContinuousTensorSpec(
                shape=torch.Size([self.outdim]), device=device
            )

        return observation_spec

    @staticmethod
    def _load_weights(model_name, vip_instance, dir_prefix):
        if model_name not in ("vip_50"):
            raise ValueError("model_name should be 'vip_50'")
        url = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
        d = load_state_dict_from_url(
            url,
            progress=True,
            map_location=next(vip_instance.parameters()).device,
            model_dir=dir_prefix,
        )
        td = TensorDict(d["vip"], []).unflatten_keys(".")
        td_flatten = td["module"]["convnet"].flatten_keys(".")
        state_dict = td_flatten.to_dict()
        vip_instance.convnet.load_state_dict(state_dict)

    def load_weights(self, dir_prefix=None):
        self._load_weights(self.model_name, self, dir_prefix)


def _init_first(fun):
    def new_fun(self, *args, **kwargs):
        if not self.initialized:
            self._init()
        return fun(self, *args, **kwargs)

    return new_fun


class VIPTransform(Compose):
    """VIP Transform class.

    VIP provides pre-trained ResNet weights aimed at facilitating visual
    embedding and reward for robotic tasks. The models are trained using Ego4d.
    See the paper:
        VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training (Jason Ma
            Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar*, Amy Zhang*)

    Args:
        model_name (str): one of resnet50
        keys_in (list of str, optional): list of input keys. If left empty, the
            "next_pixels" key is assumed.
        keys_out (list of str, optional): list of output keys. If left empty,
             "next_vip_vec" is assumed.
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
        keys_in: List[str] = None,
        keys_out: List[str] = None,
        size: int = 244,
        stack_images: bool = True,
        download: bool = False,
        download_path: Optional[str] = None,
        tensor_pixels_keys: List[str] = None,
    ):
        super().__init__()
        self.keys_in = keys_in
        self.download = download
        self.download_path = download_path
        self.model_name = model_name
        self.keys_out = keys_out
        self.size = size
        self.stack_images = stack_images
        self.tensor_pixels_keys = tensor_pixels_keys

    def _init(self):
        keys_in = self.keys_in
        model_name = self.model_name
        keys_out = self.keys_out
        size = self.size
        stack_images = self.stack_images
        tensor_pixels_keys = self.tensor_pixels_keys

        # ToTensor
        transforms = []
        if tensor_pixels_keys:
            for i in range(len(keys_in)):
                transforms.append(
                    CatTensors(
                        keys_in=[keys_in[i]],
                        out_key=tensor_pixels_keys[i],
                        del_keys=False,
                    )
                )

        totensor = ToTensorImage(
            unsqueeze=False,
            keys_in=keys_in,
        )
        transforms.append(totensor)

        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = ObservationNorm(
            keys_in=keys_in,
            loc=torch.tensor(mean).view(3, 1, 1),
            scale=torch.tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        transforms.append(normalize)

        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size, keys_in=keys_in)
        transforms.append(resize)

        # VIP
        if keys_out is None:
            if stack_images:
                keys_out = ["next_vip_vec"]
            else:
                keys_out = [f"next_vip_vec_{i}" for i in range(len(keys_in))]
        elif stack_images and len(keys_out) != 1:
            raise ValueError(
                f"key_out must be of length 1 if stack_images is True. Got keys_out={keys_out}"
            )
        elif not stack_images and len(keys_out) != len(keys_in):
            raise ValueError(
                "key_out must be of length equal to keys_in if stack_images is False."
            )

        if stack_images and len(keys_in) > 1:
            if self.is_3d:
                unsqueeze = UnsqueezeTransform(
                    keys_in=keys_in,
                    keys_out=keys_in,
                    unsqueeze_dim=-4,
                )
                transforms.append(unsqueeze)

            cattensors = CatTensors(
                keys_in,
                keys_out[0],
                dim=-4,
            )
            network = _VIPNet(
                in_keys=keys_out,
                out_keys=keys_out,
                model_name=model_name,
                del_keys=False,
            )
            flatten = FlattenObservation(-2, -1, keys_out)
            transforms = [*transforms, cattensors, network, flatten]
        else:
            network = _VIPNet(
                in_keys=keys_in,
                out_keys=keys_out,
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
