import importlib
from typing import Union

import torch

from torchrl.data import (
    CompositeSpec,
    DEVICE_TYPING,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import Transform

_has_vc = importlib.util.find_spec("vc_models")


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
        >>> transform = VC1Transform("resnet50", in_keys=["pixels"])
        >>> env.append_transform(transform)
        >>> # the forward method will first call _init which will look at env.observation_spec
        >>> env.reset()

    Args:
        model_name (str): one of resnet50, resnet34 or resnet18
        in_keys (list of str): list of input keys. If left empty, the
            "pixels" key is assumed.
        out_keys (list of str, optional): list of output keys. If left empty,
             "VC1_vec" is assumed.
        stack_images (bool, optional): if False, the images given in the :obj:`in_keys`
             argument will be treaded separetely and each will be given a single,
             separated entry in the output tensordict. Defaults to ``True``.
        download (bool, torchvision Weights config or corresponding string):
            if ``True``, the weights will be downloaded using the torch.hub download
            API (i.e. weights will be cached for future use).
            These weights are the original weights from the VC1 publication.
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

    inplace = False

    def __init__(self, in_keys, out_keys, model_name=None, del_keys: bool = True):

        from vc_models.models.vit import model_utils

        if model_name is None:
            model_name = model_utils.VC1_LARGE_NAME
        self.model_name = model_name
        self.del_keys = del_keys

        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _init(self):
        from vc_models.models.vit import model_utils

        model, embd_size, model_transforms, model_info = model_utils.load_model(
            self.model_name
        )
        self.model = model
        self.embd_size = embd_size
        self.model_transforms = model_transforms

    def _call(self, tensordict):
        tensordict_view = tensordict.view(-1)
        super()._call(tensordict_view)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict

    forward = _call

    @torch.no_grad()
    def _apply_transform(self, obs: torch.Tensor) -> None:
        shape = None
        if obs.ndimension() > 4:
            shape = obs.shape[:-3]
            obs = obs.flatten(0, -4)
        obs = self.model_transforms(obs)
        out = self.model(obs)
        if shape is not None:
            out = out.view(*shape, *out.shape[1:])
        return out

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError("_VC1Net can only infer CompositeSpec")

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]
        device = observation_spec[keys[0]].device
        dim = observation_spec[keys[0]].shape[:-3]

        observation_spec = CompositeSpec(observation_spec, shape=observation_spec.shape)
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
