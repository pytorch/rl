# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import torch
from tensordict import set_lazy_legacy, TensorDict, TensorDictBase
from torch.hub import load_state_dict_from_url

from torchrl.data.tensor_specs import Composite, TensorSpec, Unbounded
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
from torchrl.envs.transforms.utils import _set_missing_tolerance

_has_tv = importlib.util.find_spec("torchvision", None) is not None


VIP_MODEL_MAP = {
    "resnet50": "vip_50",
}


class _VIPNet(Transform):

    inplace = False

    def __init__(self, in_keys, out_keys, model_name="resnet50", del_keys: bool = True):
        if not _has_tv:
            raise ImportError(
                "Tried to instantiate VIP without torchvision. Make sure you have "
                "torchvision installed in your environment."
            )

        from torchvision import models

        self.model_name = model_name
        if model_name == "resnet50":
            self.outdim = 2048
            convnet = models.resnet50()
            convnet.fc = torch.nn.Linear(self.outdim, 1024)
        else:
            raise NotImplementedError(
                f"model {model_name} is currently not supported by VIP"
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.convnet = convnet
        self.del_keys = del_keys

    @set_lazy_legacy(False)
    def _call(self, next_tensordict):
        with next_tensordict.view(-1) as tensordict_view:
            super()._call(tensordict_view)

        if self.del_keys:
            next_tensordict.exclude(*self.in_keys, inplace=True)
        return next_tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: Check this makes sense
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

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
        if not isinstance(observation_spec, Composite):
            raise ValueError("_VIPNet can only infer Composite")

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]
        device = observation_spec[keys[0]].device
        dim = observation_spec[keys[0]].shape[:-3]

        observation_spec = observation_spec.clone()
        if self.del_keys:
            for in_key in keys:
                del observation_spec[in_key]

        for out_key in self.out_keys:
            observation_spec[out_key] = Unbounded(
                shape=torch.Size([*dim, 1024]), device=device
            )

        return observation_spec

    @staticmethod
    def _load_weights(model_name, vip_instance, dir_prefix):
        if model_name not in ("vip_50",):
            raise ValueError(f"model_name should be 'vip_50', got {model_name}")
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

    def load_weights(self, dir_prefix=None, tv_weights=None):
        from torchvision import models
        from torchvision.models import ResNet50_Weights

        if dir_prefix is not None and tv_weights is not None:
            raise RuntimeError(
                "torchvision weights API does not allow for custom download path."
            )
        elif tv_weights is not None:
            model_name = self.model_name
            if model_name == "resnet50":
                if isinstance(tv_weights, str):
                    tv_weights = getattr(ResNet50_Weights, tv_weights)
                convnet = models.resnet50(weights=tv_weights)
            else:
                raise NotImplementedError(
                    f"model {model_name} is currently not supported by R3M"
                )
            convnet.fc = torch.nn.Linear(self.outdim, 1024)
            self.convnet.load_state_dict(convnet.state_dict())

        else:
            model_name = VIP_MODEL_MAP[self.model_name]
            self._load_weights(model_name, self, dir_prefix)


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
        in_keys (list of str, optional): list of input keys. If left empty, the
            "pixels" key is assumed.
        out_keys (list of str, optional): list of output keys. If left empty,
             "vip_vec" is assumed.
        size (int, optional): Size of the image to feed to resnet.
            Defaults to 244.
        stack_images (bool, optional): if False, the images given in the :obj:`in_keys`
             argument will be treaded separately and each will be given a single,
             separated entry in the output tensordict. Defaults to ``True``.
        download (bool, torchvision Weights config or corresponding string):
            if ``True``, the weights will be downloaded using the torch.hub download
            API (i.e. weights will be cached for future use).
            These weights are the original weights from the VIP publication.
            If the torchvision weights are needed, there are two ways they can be
            obtained: :obj:`download=ResNet50_Weights.IMAGENET1K_V1` or :obj:`download="IMAGENET1K_V1"`
            where :obj:`ResNet50_Weights` can be imported via :obj:`from torchvision.models import resnet50, ResNet50_Weights`.
            Defaults to False.
        download_path (str, optional): path where to download the models.
            Default is None (cache path determined by torch.hub utils).
        tensor_pixels_keys (list of str, optional): Optionally, one can keep the
            original images (as collected from the env) in the output tensordict.
            If no value is provided, this won't be collected.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls.initialized = False
        cls._device = None
        cls._dtype = None
        return super().__new__(cls)

    def __init__(
        self,
        model_name: str,
        in_keys: list[str] = None,
        out_keys: list[str] = None,
        size: int = 244,
        stack_images: bool = True,
        download: bool | WeightsEnum | str = False,  # noqa: F821
        download_path: str | None = None,
        tensor_pixels_keys: list[str] = None,
    ):
        super().__init__()
        self.in_keys = in_keys if in_keys is not None else ["pixels"]
        self.download = download
        self.download_path = download_path
        self.model_name = model_name
        self.out_keys = out_keys
        self.size = size
        self.stack_images = stack_images
        self.tensor_pixels_keys = tensor_pixels_keys
        self._init()

    def _init(self):
        """Initializer for VIP."""
        self.initialized = True
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
            loc=torch.as_tensor(mean).view(3, 1, 1),
            scale=torch.as_tensor(std).view(3, 1, 1),
            standard_normal=True,
        )
        transforms.append(normalize)

        # Resize: note that resize is a no-op if the tensor has the desired size already
        resize = Resize(size, size, in_keys=in_keys)
        transforms.append(resize)

        # VIP
        if out_keys in (None, []):
            if stack_images:
                out_keys = ["vip_vec"]
            else:
                out_keys = [f"vip_vec_{i}" for i in range(len(in_keys))]
            self.out_keys = out_keys
        elif stack_images and len(out_keys) != 1:
            raise ValueError(
                f"out_key must be of length 1 if stack_images is True. Got out_keys={out_keys}"
            )
        elif not stack_images and len(out_keys) != len(in_keys):
            raise ValueError(
                "out_key must be of length equal to in_keys if stack_images is False."
            )

        if stack_images and len(in_keys) > 1:
            unsqueeze = UnsqueezeTransform(
                in_keys=in_keys,
                out_keys=in_keys,
                dim=-4,
            )
            transforms.append(unsqueeze)

            cattensors = CatTensors(
                in_keys,
                out_keys[0],
                dim=-4,
            )
            network = _VIPNet(
                in_keys=out_keys,
                out_keys=out_keys,
                model_name=model_name,
                del_keys=False,
            )
            flatten = FlattenObservation(-2, -1, out_keys)
            transforms = [*transforms, cattensors, network, flatten]
        else:
            network = _VIPNet(
                in_keys=in_keys,
                out_keys=out_keys,
                model_name=model_name,
                del_keys=True,
            )
            transforms = [*transforms, network]

        for transform in transforms:
            self.append(transform)
        if self.download is True:
            self[-1].load_weights(dir_prefix=self.download_path, tv_weights=None)
        elif self.download:
            self[-1].load_weights(
                dir_prefix=self.download_path, tv_weights=self.download
            )

        if self._device is not None:
            self.to(self._device)
        if self._dtype is not None:
            self.to(self._dtype)

    def to(self, dest: DEVICE_TYPING | torch.dtype):
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


class VIPRewardTransform(VIPTransform):
    """A VIP transform to compute rewards based on embedded similarity.

    This class will update the reward computation
    """

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if "goal_embedding" not in tensordict.keys():
            tensordict = self._embed_goal(tensordict)
        tensordict_reset.set("goal_embedding", tensordict.pop("goal_embedding"))
        return super()._reset(tensordict, tensordict_reset)

    def _embed_goal(self, tensordict):
        if "goal_image" not in tensordict.keys():
            raise KeyError(
                f"{self.__class__.__name__}.reset() requires a `'goal_image'` key to be "
                f"present in the input tensordict. Got keys {list(tensordict.keys())}."
            )
        tensordict_in = tensordict.select("goal_image").rename_key_(
            "goal_image", self.in_keys[0]
        )
        tensordict_in = super().forward(tensordict_in)
        tensordict = tensordict.update(
            tensordict_in.rename_key_(self.out_keys[0], "goal_embedding")
        )
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if "goal_embedding" not in tensordict.keys():
            tensordict = self._embed_goal(tensordict)
        last_embedding_key = self.out_keys[0]
        last_embedding = tensordict.get(last_embedding_key, None)
        next_tensordict = super()._step(tensordict, next_tensordict)
        cur_embedding = next_tensordict.get(self.out_keys[0])
        if last_embedding is not None:
            goal_embedding = tensordict["goal_embedding"]
            reward = -torch.linalg.norm(cur_embedding - goal_embedding, dim=-1) - (
                -torch.linalg.norm(last_embedding - goal_embedding, dim=-1)
            )
            next_tensordict.set("reward", reward)
        return next_tensordict

    def forward(self, tensordict):
        tensordict = super().forward(tensordict)
        return tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        if "full_state_spec" in input_spec.keys():
            full_state_spec = input_spec["full_state_spec"]
        else:
            full_state_spec = Composite(
                shape=input_spec.shape, device=input_spec.device
            )
        # find the obs spec
        in_key = self.in_keys[0]
        spec = self.parent.output_spec["full_observation_spec"][in_key]
        full_state_spec["goal_image"] = spec.clone()
        input_spec["full_state_spec"] = full_state_spec
        return super().transform_input_spec(input_spec)
