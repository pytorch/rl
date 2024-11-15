# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
import warnings

from copy import copy
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs.libs.gym import (
    _AsyncMeta,
    _gym_to_torchrl_spec_transform,
    gym_backend,
    GymEnv,
)
from torchrl.envs.utils import _classproperty, make_composite_from_td

_has_gym = (
    importlib.util.find_spec("gym") is not None
    or importlib.util.find_spec("gymnasium") is not None
)
_has_robohive = importlib.util.find_spec("robohive") is not None and _has_gym

if _has_robohive:
    os.environ.setdefault("sim_backend", "MUJOCO")


class set_directory(object):
    """Sets the cwd within the context.

    Args:
        path (Path): The path to the cwd
    """

    def __init__(self, path: Path):
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args, **kwargs):
        os.chdir(self.origin)

    def __call__(self, fun):
        def new_fun(*args, **kwargs):
            with set_directory(Path(self.path)):
                return fun(*args, **kwargs)

        return new_fun


class _RoboHiveBuild(_AsyncMeta):
    def __call__(self, *args, **kwargs):
        instance: RoboHiveEnv = super().__call__(*args, **kwargs)
        instance._refine_specs()
        return instance


class RoboHiveEnv(GymEnv, metaclass=_RoboHiveBuild):
    """A wrapper for RoboHive gym environments.

    RoboHive is a collection of environments/tasks simulated with the MuJoCo physics engine exposed using the OpenAI-Gym API.

    Github: https://github.com/vikashplus/robohive/

    Doc: https://github.com/vikashplus/robohive/wiki

    Paper: https://arxiv.org/abs/2310.06828

    .. warning::
        RoboHive requires gym 0.13.

    Args:
        env_name (str): the environment name to build. Must be one of :attr:`.available_envs`
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.

    Keyword Args:
        from_pixels (bool, optional): if ``True``, an attempt to return the pixel
            observations from the env will be performed. By default, these observations
            will be written under the ``"pixels"`` entry.
            The method being used varies
            depending on the gym version and may involve a ``wrappers.pixel_observation.PixelObservationWrapper``.
            Defaults to ``False``.
        pixels_only (bool, optional): if ``True``, only the pixel observations will
            be returned (by default under the ``"pixels"`` entry in the output tensordict).
            If ``False``, observations (eg, states) and pixels will be returned
            whenever ``from_pixels=True``. Defaults to ``True``.
        from_depths (bool, optional): if ``True``, an attempt to return the depth
            observations from the env will be performed. By default, these observations
            will be written under the ``"depths"`` entry. Requires ``from_pixels`` to be ``True``.
            Defaults to ``False``.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): Only ``torch.Size([])`` will work with
            ``RoboHiveEnv`` since vectorized environments are not supported within the
            class. To execute more than one environment at a time, see :class:`~torchrl.envs.ParallelEnv`.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs (list): a list of available envs to build.

    Examples:
        >>> from torchrl.envs import RoboHiveEnv
        >>> env = RoboHiveEnv(RoboHiveEnv.available_envs[0])
        >>> env.rollout(3)

    """

    env_list = []

    @_classproperty
    def CURR_DIR(cls):
        if _has_robohive:
            import robohive.envs.multi_task.substeps1

            return robohive.envs.multi_task.substeps1.CURR_DIR
        else:
            return None

    @_classproperty
    def available_envs(cls):
        if not _has_robohive:
            return []
        cls.register_envs()
        return cls.env_list

    @classmethod
    def register_envs(cls):
        if not _has_robohive:
            raise ImportError(
                "Cannot load robohive from the current virtual environment."
            )
        from robohive import robohive_env_suite as robohive_envs
        from robohive.utils.prompt_utils import Prompt, set_prompt_verbosity

        set_prompt_verbosity(Prompt.WARN)
        cls.env_list += robohive_envs
        if not len(robohive_envs):
            raise RuntimeError("did not load any environment.")

    def _build_env(  # noqa: F811
        self,
        env_name: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
        from_depths: bool = False,
        **kwargs,
    ) -> "gym.core.Env":  # noqa: F821
        if from_pixels:
            if "cameras" not in kwargs:
                warnings.warn(
                    "from_pixels=True will lead to a registration of ALL available cameras, "
                    "which may lead to performance issue. "
                    "Consider passing only the needed cameras through cameras=list_of_cameras. "
                    "The list of available cameras for a specific environment can be obtained via "
                    "RobohiveEnv.get_available_cams(env_name)."
                )
                kwargs["cameras"] = self.get_available_cams(env_name)
            cams = list(kwargs.pop("cameras"))
            env_name = self.register_visual_env(
                cams=cams, env_name=env_name, from_depths=from_depths
            )

        elif "cameras" in kwargs and kwargs["cameras"]:
            raise RuntimeError("Got a list of cameras but from_pixels is set to False.")

        self.pixels_only = pixels_only
        try:
            render_device = int(str(self.device)[-1])
        except ValueError:
            render_device = 0

        if not _has_robohive:
            raise ImportError(
                f"gym/robohive not found, unable to create {env_name}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        try:
            env = self.lib.make(
                env_name,
                frameskip=self.frame_skip,
                device_id=render_device,
                return_dict=True,
                **kwargs,
            )
            self.wrapper_frame_skip = 1
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise err
            kwargs.pop("framek_skip")
            env = self.lib.make(
                env_name, return_dict=True, device_id=render_device, **kwargs
            )
            self.wrapper_frame_skip = self.frame_skip
        # except Exception as err:
        #     raise RuntimeError(f"Failed to build env {env_name}.") from err
        self.from_pixels = from_pixels
        self.from_depths = from_depths
        self.render_device = render_device
        if kwargs.get("read_info", True):
            self.set_info_dict_reader(self.read_info)
        return env

    def _make_specs(self, env: "gym.Env", batch_size=None) -> None:  # noqa: F821
        out = super()._make_specs(env=env, batch_size=batch_size)
        self.env.reset()
        *_, info = self.env.step(self.env.action_space.sample())
        info = self.read_info(info, TensorDict())
        info = info.get("info")
        self.observation_spec["info"] = make_composite_from_td(info)
        return out

    @classmethod
    def register_visual_env(cls, env_name, cams, from_depths):
        with set_directory(cls.CURR_DIR):
            from robohive.envs.env_variants import register_env_variant

            if not len(cams):
                raise RuntimeError("Cannot create a visual envs without cameras.")
            cams = sorted(cams)
            cams_rep = [i.replace("A:", "A_") for i in cams]
            new_env_name = "-".join([cam[:-3] for cam in cams_rep] + [env_name])
            visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
            if from_depths:
                visual_keys.extend([f"d:{c}:224x224:2d" for c in cams])
            register_env_variant(
                env_name,
                variants={
                    "visual_keys": visual_keys,
                },
                variant_id=new_env_name,
            )
            env_name = new_env_name
            cls.env_list += [env_name]
            return env_name

    def _refine_specs(self) -> None:  # noqa: F821
        env = self._env
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space, device=self.device
        )
        # get a np rollout
        rollout = TensorDict({"done": torch.zeros(3, 1)}, [3])
        env.reset()

        def get_obs():
            _dict = {}
            obs_dict = copy(env.obs_dict)
            if self.from_pixels:
                visual = self.env.get_exteroception()
                obs_dict.update(visual)
            pixel_list, depth_list = [], []
            for obs_key in obs_dict:
                if obs_key.startswith("rgb"):
                    pix = obs_dict[obs_key]
                    if not pix.shape[0] == 1:
                        pix = pix[None]
                    pixel_list.append(pix)
                elif obs_key.startswith("d:"):
                    dep = obs_dict[obs_key]
                    dep = dep[None]
                    depth_list.append(dep)
                elif obs_key in env.obs_keys:
                    value = env.obs_dict[obs_key]
                    if not value.shape:
                        value = value[None]
                    _dict[obs_key] = value
            if pixel_list:
                _dict["pixels"] = np.concatenate(pixel_list, 0)
            if depth_list:
                _dict["depths"] = np.concatenate(depth_list, 0)
            return _dict

        for i in range(3):
            _dict = {}
            _dict.update(get_obs())
            _dict["action"] = action = env.action_space.sample()
            _, r, trunc, term, done, _ = self._output_transform(env.step(action))
            _dict[("next", "reward")] = r.reshape(1)
            _dict[("next", "done")] = [1]
            _dict[("next", "terminated")] = [1]
            _dict[("next", "truncated")] = [1]
            _dict["next"] = get_obs()
            rollout[i] = TensorDict(_dict, [])

        observation_spec = make_composite_from_td(
            rollout.get("next").exclude("done", "reward", "terminated", "truncated")[0]
        )
        self.observation_spec = observation_spec

        self.reward_spec = Unbounded(
            shape=(1,),
            device=self.device,
        )  # default

        rollout = self.rollout(2, return_contiguous=False).get("next")
        rollout = rollout.exclude(
            self.reward_key, *self.done_keys, *self.observation_spec.keys(True, True)
        )
        rollout = rollout[..., 0]
        spec = make_composite_from_td(rollout)
        self.observation_spec.update(spec)
        self.empty_cache()

    def _reset_output_transform(self, reset_data):
        if not (isinstance(reset_data, tuple) and len(reset_data) == 2):
            return reset_data, {}
        return reset_data

    def set_from_pixels(self, from_pixels: bool) -> None:
        """Sets the from_pixels attribute to an existing environment.

        Args:
            from_pixels (bool): new value for the from_pixels attribute

        """
        if from_pixels is self.from_pixels:
            return
        self.from_pixels = from_pixels
        self._refine_specs()

    def read_obs(self, observation):
        # the info is missing from the reset
        observations = self.env.obs_dict
        try:
            del observations["t"]
        except KeyError:
            pass
        # recover vec
        obsdict = {}
        pixel_list, depth_list = [], []
        if self.from_pixels:
            visual = self.env.get_exteroception()
            observations.update(visual)
        for key in observations:
            if key.startswith("rgb"):
                pix = observations[key]
                if not pix.shape[0] == 1:
                    pix = pix[None]
                pixel_list.append(pix)
            elif key.startswith("d:"):
                dep = observations[key]
                dep = dep[None]
                depth_list.append(dep)
            elif key in self._env.obs_keys:
                value = observations[key]
                if not value.shape:
                    value = value[None]
                obsdict[key] = value  # ravel helps with images
        # if obsvec:
        #     obsvec = np.concatenate(obsvec, 0)
        if self.from_pixels:
            obsdict.update({"pixels": np.concatenate(pixel_list, 0)})
        if self.from_pixels and self.from_depths:
            obsdict.update({"depths": np.concatenate(depth_list, 0)})
        out = obsdict
        return super().read_obs(out)

    def read_info(self, info, tensordict_out):
        if not info:
            info_spec = self.observation_spec.get("info", None)
            if info_spec is None:
                return tensordict_out
            tensordict_out.set("info", info_spec.zero())
            return tensordict_out
        out = (
            TensorDict(info, [])
            .filter_non_tensor_data()
            .exclude("obs_dict", "done", "reward", *self._env.obs_keys, "act")
            .apply(lambda x: x, filter_empty=True)
        )
        if "info" in self.observation_spec.keys():
            info_spec = self.observation_spec["info"]

            def func(name, x):
                spec = info_spec.get(name, None)
                if spec is None:
                    return None
                return x.reshape(info_spec[name].shape)

            out.update(out.named_apply(func, nested_keys=True, filter_empty=True))
        else:
            out.update(
                out.apply(
                    lambda x: x.reshape((1,)) if not x.shape else x, filter_empty=True
                )
            )
        tensordict_out.set("info", out)
        return tensordict_out

    def _init_env(self):
        pass

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        try:
            render_device = int(str(out.device)[-1])
        except ValueError:
            render_device = 0
        if render_device != self.render_device:
            out._build_env(**self._constructor_kwargs)
        return out

    @classmethod
    def get_available_cams(cls, env_name):
        env = gym_backend().make(env_name)
        cams = [env.sim.model.id2name(ic, 7) for ic in range(env.sim.model.ncam)]
        return cams
