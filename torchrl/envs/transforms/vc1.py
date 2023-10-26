# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import subprocess
from functools import partial
from typing import Union

import torch
from tensordict import TensorDictBase
from torch import nn

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DEVICE_TYPING,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.transforms.transforms import (
    CenterCrop,
    Compose,
    ObservationNorm,
    Resize,
    ToTensorImage,
    Transform,
)
from torchrl.envs.transforms.utils import _set_missing_tolerance

_has_vc = importlib.util.find_spec("vc_models") is not None


class VC1Transform(Transform):
    """VC1 Transform class.

    VC1 provides pre-trained ResNet weights aimed at facilitating visual
    embedding for robotic tasks. The models are trained using Ego4d.

    See the paper:
        VC1: A Universal Visual Representation for Robot Manipulation (Suraj Nair,
            Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta)
            https://arxiv.org/abs/2203.12601

    The VC1Transform is created in a lazy manner: the object will be initialized
    only when an attribute (a spec or the forward method) will be queried.
    The reason for this is that the :obj:`_init()` method requires some attributes of
    the parent environment (if any) to be accessed: by making the class lazy we
    can ensure that the following code snippet works as expected:

    Examples:
        >>> transform = VC1Transform("default", in_keys=["pixels"])
        >>> env.append_transform(transform)
        >>> # the forward method will first call _init which will look at env.observation_spec
        >>> env.reset()

    Args:
        in_keys (list of NestedKeys): list of input keys. If left empty, the
            "pixels" key is assumed.
        out_keys (list of NestedKeys, optional): list of output keys. If left empty,
             "VC1_vec" is assumed.
        model_name (str): One of ``"large"``, ``"base"`` or any other compatible
            model name (see the `github repo <https://github.com/facebookresearch/eai-vc>`_ for more info). Defaults to ``"default"``
            which provides a small, untrained model for testing.
        del_keys (bool, optional): If ``True`` (default), the input key will be
            discarded from the returned tensordict.
    """

    inplace = False
    IMPORT_ERROR = (
        "Could not load vc_models. You can install it via "
        "VC1Transform.install_vc_models()."
    )

    def __init__(self, in_keys, out_keys, model_name, del_keys: bool = True):
        if model_name == "default":
            self.make_noload_model()
            model_name = "vc1_vitb_noload"
        self.model_name = model_name
        self.del_keys = del_keys

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._init()

    def _init(self):
        try:
            from vc_models.models.vit import model_utils
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(self.IMPORT_ERROR) from err

        if self.model_name == "base":
            model_name = model_utils.VC1_BASE_NAME
        elif self.model_name == "large":
            model_name = model_utils.VC1_LARGE_NAME
        else:
            model_name = self.model_name

        model, embd_size, model_transforms, model_info = model_utils.load_model(
            model_name
        )
        self.model = model
        self.embd_size = embd_size
        self.model_transforms = self._map_tv_to_torchrl(model_transforms)

    def _map_tv_to_torchrl(
        self,
        model_transforms,
        in_keys=None,
    ):
        if in_keys is None:
            in_keys = self.in_keys
        from torchvision import transforms

        if isinstance(model_transforms, transforms.Resize):
            size = model_transforms.size
            if isinstance(size, int):
                size = (size, size)
            return Resize(
                *size,
                in_keys=in_keys,
            )
        elif isinstance(model_transforms, transforms.CenterCrop):
            size = model_transforms.size
            if isinstance(size, int):
                size = (size,)
            return CenterCrop(
                *size,
                in_keys=in_keys,
            )
        elif isinstance(model_transforms, transforms.Normalize):
            return ObservationNorm(
                in_keys=in_keys,
                loc=torch.tensor(model_transforms.mean).reshape(3, 1, 1),
                scale=torch.tensor(model_transforms.std).reshape(3, 1, 1),
                standard_normal=True,
            )
        elif isinstance(model_transforms, transforms.ToTensor):
            return ToTensorImage(
                in_keys=in_keys,
            )
        elif isinstance(model_transforms, transforms.Compose):
            transform_list = []
            for t in model_transforms.transforms:

                if isinstance(t, transforms.ToTensor):
                    transform_list.insert(0, t)
                else:
                    transform_list.append(t)
            if len(transform_list) == 0:
                raise RuntimeError("Did not find any transform.")
            for i, t in enumerate(transform_list):
                if i == 0:
                    transform_list[i] = self._map_tv_to_torchrl(t)
                else:
                    transform_list[i] = self._map_tv_to_torchrl(t)
            return Compose(*transform_list)
        else:
            raise NotImplementedError(type(model_transforms))

    def _call(self, tensordict):
        if not self.del_keys:
            in_keys = [
                in_key
                for in_key, out_key in zip(self.in_keys, self.out_keys)
                if in_key != out_key
            ]
            saved_td = tensordict.select(*in_keys)
        tensordict_view = tensordict.view(-1)
        super()._call(self.model_transforms(tensordict_view))
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        else:
            # reset in_keys
            tensordict.update(saved_td)
        return tensordict

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
        out = self.model(obs)
        if shape is not None:
            out = out.view(*shape, *out.shape[1:])
        return out

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError("VC1Transform can only infer CompositeSpec")

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]
        device = observation_spec[keys[0]].device
        dim = observation_spec[keys[0]].shape[:-3]

        observation_spec = observation_spec.clone()
        if self.del_keys:
            for in_key in keys:
                del observation_spec[in_key]

        for out_key in self.out_keys:
            observation_spec[out_key] = UnboundedContinuousTensorSpec(
                shape=torch.Size([*dim, self.embd_size]), device=device
            )

        return observation_spec

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

    @classmethod
    def install_vc_models(cls, auto_exit=False):
        try:
            from vc_models import models  # noqa: F401

            print("vc_models found, no need to install.")
        except ModuleNotFoundError:
            HOME = os.environ.get("HOME")
            vcdir = HOME + "/.cache/torchrl/eai-vc"
            parentdir = os.path.dirname(os.path.abspath(vcdir))
            print(parentdir)
            os.makedirs(parentdir, exist_ok=True)
            try:
                from git import Repo
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    "Could not load git. Make sure that `git` has been installed "
                    "in your virtual environment."
                ) from err
            Repo.clone_from("https://github.com/facebookresearch/eai-vc.git", vcdir)
            os.chdir(vcdir + "/vc_models")
            subprocess.call(["python", "setup.py", "develop"])
            if not auto_exit:
                input(
                    "VC1 has been successfully installed. Exit this python run and "
                    "relaunch it again. Press Enter to exit..."
                )
                exit()

    @classmethod
    def make_noload_model(cls):
        """Creates an naive model at a custom destination."""
        import vc_models

        models_filepath = os.path.dirname(os.path.abspath(vc_models.__file__))
        cfg_path = os.path.join(
            models_filepath, "conf", "model", "vc1_vitb_noload.yaml"
        )
        if os.path.exists(cfg_path):
            return
        config = """_target_: vc_models.models.load_model
model:
  _target_: vc_models.models.vit.vit.load_mae_encoder
  checkpoint_path:
  model:
    _target_: torchrl.envs.transforms.vc1._vit_base_patch16
    img_size: 224
    use_cls: True
    drop_path_rate: 0.0
transform:
  _target_: vc_models.transforms.vit_transforms
metadata:
  algo: mae
  model: vit_base_patch16
  data:
    - ego
    - imagenet
    - inav
  comment: 182_epochs
"""
        with open(cfg_path, "w") as file:
            file.write(config)


def _vit_base_patch16(**kwargs):
    from vc_models.models.vit.vit import VisionTransformer

    model = VisionTransformer(
        patch_size=16,
        embed_dim=16,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
