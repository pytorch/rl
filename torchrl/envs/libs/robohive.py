# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import make_tensordict
from torchrl._utils import implement_for
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.libs.gym import (
    _gym_to_torchrl_spec_transform,
    GymEnv,
    set_gym_backend,
)
from torchrl.envs.utils import make_composite_from_td

_has_robohive = importlib.util.find_spec("robohive") is not None

if _has_robohive:
    os.environ.setdefault("sim_backend", "MUJOCO")
    import gym

    with set_gym_backend("gym"):
        existing_envs = set(GymEnv.available_envs)
    import robohive.envs.multi_task.substeps1
    from robohive.envs.env_variants import register_env_variant


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


class RoboHiveEnv(GymEnv):
    """A wrapper for RoboHive gym environments.

    RoboHive is a collection of environments/tasks simulated with the MuJoCo physics engine exposed using the OpenAI-Gym API.

    Github: https://github.com/vikashplus/robohive/

    RoboHive requires gym 0.13
    """

    env_list = []
    if _has_robohive:
        CURR_DIR = robohive.envs.multi_task.substeps1.CURR_DIR
    else:
        CURR_DIR = None

    @classmethod
    def register_envs(cls):
        if not _has_robohive:
            raise ImportError("Cannot load robohive.")
        with set_gym_backend("gym"):
            robo_envs = set(GymEnv.available_envs) - existing_envs
        cls.env_list += robo_envs
        if not len(robo_envs):
            raise RuntimeError("did not load any environment.")

    @implement_for(
        "gym", "0.14", None
    )  # make sure gym 0.13 is installed, otherwise raise an exception
    def _build_env(self, *args, **kwargs):
        raise NotImplementedError(
            "Your gym version is too recent, RoboHiveEnv is only compatible with gym 0.13."
        )

    @implement_for(
        "gym", "0.13", "0.14"
    )  # make sure gym 0.13 is installed, otherwise raise an exception
    def _build_env(  # noqa: F811
        self,
        env_name: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
        **kwargs,
    ) -> "gym.core.Env":

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
            env_name = self.register_visual_env(cams=cams, env_name=env_name)

        elif "cameras" in kwargs and kwargs["cameras"]:
            raise RuntimeError("Got a list of cameras but from_pixels is set to False.")

        self.pixels_only = pixels_only
        try:
            render_device = int(str(self.device)[-1])
        except ValueError:
            render_device = 0
        print(f"rendering device: {render_device}, device is {self.device}")

        if not _has_robohive:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
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
            if env.visual_keys:
                from_pixels = bool(len(env.visual_keys))
            else:
                from_pixels = False
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise err
            kwargs.pop("framek_skip")
            env = self.lib.make(
                env_name, return_dict=True, device_id=render_device, **kwargs
            )
            self.wrapper_frame_skip = self.frame_skip

        self.from_pixels = from_pixels
        self.render_device = render_device
        self.info_dict_reader = self.read_info
        return env

    @classmethod
    def register_visual_env(cls, env_name, cams):
        with set_directory(cls.CURR_DIR):
            if not len(cams):
                raise RuntimeError("Cannot create a visual envs without cameras.")
            cams = sorted(cams)
            print("cams", cams)
            new_env_name = "-".join([cam[:-3] for cam in cams] + [env_name])
            print("new_env_name", new_env_name)
            if new_env_name in cls.env_list:
                return new_env_name
            visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
            print(f"env visual_keys: {visual_keys}")
            print(f"new env name: {new_env_name}")
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

    def _make_specs(self, env: "gym.Env") -> None:
        if self.from_pixels:
            num_cams = len(env.visual_keys)
            # n_pix = 224 * 224 * 3 * num_cams
            # env.observation_space = gym.spaces.Box(
            #     -8 * np.ones(env.obs_dim - n_pix),
            #     8 * np.ones(env.obs_dim - n_pix),
            #     dtype=np.float32,
            # )
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space, device=self.device
        )
        observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
        )
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(observation=observation_spec)
        self.observation_spec = observation_spec
        if self.from_pixels:
            self.observation_spec["pixels"] = BoundedTensorSpec(
                torch.zeros(
                    num_cams,
                    224,  # working with 640
                    224,  # working with 480
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                255
                * torch.ones(
                    num_cams,
                    224,
                    224,
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                torch.Size(torch.Size([num_cams, 224, 224, 3])),
                dtype=torch.uint8,
                device=self.device,
            )

        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )  # default

        rollout = self.rollout(2).get("next").exclude("done", "reward")[0]
        self.observation_spec.update(make_composite_from_td(rollout))

    def set_from_pixels(self, from_pixels: bool) -> None:
        """Sets the from_pixels attribute to an existing environment.

        Args:
            from_pixels (bool): new value for the from_pixels attribute

        """
        if from_pixels is self.from_pixels:
            return
        self.from_pixels = from_pixels
        self._make_specs(self.env)

    def read_obs(self, observation):
        # the info is missing from the reset
        observations = self.env.obs_dict
        try:
            del observations["t"]
        except KeyError:
            pass
        # recover vec
        obsvec = []
        pixel_list = []
        if self.from_pixels:
            visual = self.env.get_exteroception()
            observations.update(visual)
        for key in observations:
            if key.startswith("rgb"):
                pix = observations[key]
                if not pix.shape[0] == 1:
                    pix = pix[None]
                pixel_list.append(pix)
            elif key in self._env.obs_keys:
                value = observations[key]
                if not value.shape:
                    value = value[None]
                obsvec.append(value)  # ravel helps with images
        if obsvec:
            obsvec = np.concatenate(obsvec, 0)
        if self.from_pixels:
            out = {"observation": obsvec, "pixels": np.concatenate(pixel_list, 0)}
        else:
            out = {"observation": obsvec}
        return super().read_obs(out)

    def read_info(self, info, tensordict_out):
        out = {}
        for key, value in info.items():
            if key in ("obs_dict", "done", "reward"):
                continue
            if isinstance(value, dict):
                value = {key: _val for key, _val in value.items() if _val is not None}
                value = make_tensordict(value, batch_size=[])
            if value is not None:
                out[key] = value
        tensordict_out.update(out)
        return tensordict_out

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
        env = gym.make(env_name)
        cams = [env.sim.model.id2name(ic, 7) for ic in range(env.sim.model.ncam)]
        print("available cameras", cams)
        return cams


if _has_robohive:
    RoboHiveEnv.register_envs()
    # MODEL_PATH = robohive.envs.multi_task.substeps1.MODEL_PATH
    # CONFIG_PATH = robohive.envs.multi_task.substeps1.CONFIG_PATH
    # RANDOM_ENTRY_POINT = robohive.envs.multi_task.substeps1.RANDOM_ENTRY_POINT
    # FIXED_ENTRY_POINT = robohive.envs.multi_task.substeps1.FIXED_ENTRY_POINT
    # ENTRY_POINT = RANDOM_ENTRY_POINT
#
#     visual_obs_keys_wt = robohive.envs.multi_task.substeps1.visual_obs_keys_wt
#
#     override_keys = [
#         "objs_jnt",
#         "end_effector",
#         "knob1_site_err",
#         "knob2_site_err",
#         "knob3_site_err",
#         "knob4_site_err",
#         "light_site_err",
#         "slide_site_err",
#         "leftdoor_site_err",
#         "rightdoor_site_err",
#         "microhandle_site_err",
#         "kettle_site0_err",
#         "rgb:right_cam:224x224:2d",
#         "rgb:left_cam:224x224:2d",
#     ]
#
#     @set_directory(CURR_DIR)
#     def register_kitchen_envs():
#         print("RLHive:> Registering Kitchen Envs")
#
#         env_list = [
#             "FK1_RelaxFixed-v4",
#             # "kitchen_close-v3",
#         ]
#
#         obs_keys_wt = {
#             "robot_jnt": 1.0,
#             "end_effector": 1.0,
#         }
#
#         env_names = copy(env_list)
#
#         for env in env_list:
#             try:
#                 _env = gym.make(env)
#                 cams = [
#                     _env.sim.model.id2name(ic, 7) for ic in range(_env.sim.model.ncam)
#                 ]
#                 if len(cams) == 0:
#                     continue
#                 visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
#                 print("env visual_keys", visual_keys)
#                 new_env_name = "visual_" + env
#                 register_env_variant(
#                     env,
#                     variants={
#                         "obs_keys_wt": obs_keys_wt,
#                         "visual_keys": visual_keys,
#                     },
#                     variant_id=new_env_name,
#                     override_keys=override_keys,
#                 )
#                 env_names += [new_env_name]
#             except AssertionError as err:
#                 warnings.warn(
#                     f"Could not register {new_env_name}, the following error was raised: {err}"
#                 )
#         return env_names
#
#     @set_directory(CURR_DIR)
#     def register_franka_envs():
#         print("RLHive:> Registering Franka Envs")
#         env_list = [
#             # "franka_slide_random-v3",
#             # "franka_slide_close-v3",
#             # "franka_slide_open-v3",
#             # "franka_micro_random-v3",
#             # "franka_micro_close-v3",
#             # "franka_micro_open-v3",
#         ]
#         env_names = copy(env_list)
#
#         # Franka Appliance ======================================================================
#         obs_keys_wt = {
#             "robot_jnt": 1.0,
#             "end_effector": 1.0,
#         }
#         # visual_obs_keys = {
#         #     "rgb:right_cam:224x224:2d": 1.0,
#         #     "rgb:left_cam:224x224:2d": 1.0,
#         # }
#         for env in env_list:
#             try:
#                 _env = gym.make(env)
#                 cams = [
#                     _env.sim.model.id2name(ic, 7) for ic in range(_env.sim.model.ncam)
#                 ]
#                 if len(cams) == 0:
#                     continue
#                 visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
#                 print("env visual_keys", visual_keys)
#                 new_env_name = "visual_" + env
#                 register_env_variant(
#                     env,
#                     variants={
#                         "obs_keys_wt": obs_keys_wt,
#                         "visual_keys": visual_keys,
#                     },
#                     variant_id=new_env_name,
#                     override_keys=override_keys,
#                 )
#                 env_names += [new_env_name]
#             except AssertionError as err:
#                 warnings.warn(
#                     f"Could not register {new_env_name}, the following error was raised: {err}"
#                 )
#         return env_names
#
#     @set_directory(CURR_DIR)
#     def _register_hand_envs():
#         print("RLHive:> Registering Arm Envs")
#         env_list = ["door-v1", "hammer-v1", "pen-v1", "relocate-v1"]
#         env_names = copy(env_list)
#
#         # Hand Manipulation Suite ======================================================================
#         for env in env_list:
#             try:
#                 _env = gym.make(env)
#                 cams = [
#                     _env.sim.model.id2name(ic, 7) for ic in range(_env.sim.model.ncam)
#                 ]
#                 if len(cams) == 0:
#                     continue
#                 visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
#                 print("env visual_keys", visual_keys)
#                 new_env_name = "visual_" + env
#                 register_env_variant(
#                     env,
#                     variants={
#                         "visual_keys": visual_keys,
#                     },
#                     variant_id=new_env_name,
#                 )
#                 env_names += [new_env_name]
#             except AssertionError as err:
#                 warnings.warn(
#                     f"Could not register {new_env_name}, the following error was raised: {err}"
#                 )
#         return env_names
#
#     @set_directory(CURR_DIR)
#     def _register_myo_envs():
#         print("RLHive:> Registering Myo Envs")
#         env_list = [
#             "motorFingerReachFixed-v0",
#             "myoFingerPoseFixed-v0",
#             "myoFingerPoseRandom-v0",
#             "myoFingerReachFixed-v0",
#             "myoFingerReachRandom-v0",
#             # "ElbowPose1D1MRandom-v0",
#             "myoElbowPose1D6MRandom-v0",
#             "myoHandPoseFixed-v0",
#             "myoHandPoseRandom-v0",
#             "myoHandReachFixed-v0",
#             "myoHandReachRandom-v0",
#             "myoHandKeyTurnFixed-v0",
#             "myoHandKeyTurnRandom-v0",
#             "myoHandObjHoldFixed-v0",
#             "myoHandObjHoldRandom-v0",
#             "myoHandPenTwirlFixed-v0",
#             "myoHandPenTwirlRandom-v0",
#             "myoChallengeDieReorientP1-v0",
#             "myoChallengeDieReorientP2-v0",
#             "myoChallengeBaodingP1-v1",
#             "myoChallengeBaodingP2-v1",
#         ]
#         env_names = copy(env_list)
#
#         visual_keys = [
#             "rgb:left_cam:224x224:2d",
#             "rgb:right_cam:224x224:2d",
#         ]
#         # Hand Manipulation Suite ======================================================================
#         for env in env_list:
#             try:
#                 _env = gym.make(env)
#                 cams = [
#                     _env.sim.model.id2name(ic, 7) for ic in range(_env.sim.model.ncam)
#                 ]
#                 if len(cams) == 0:
#                     continue
#                 visual_keys = [f"rgb:{c}:224x224:2d" for c in cams]
#                 print("env visual_keys", visual_keys)
#                 new_env_name = "visual_" + env
#                 register_env_variant(
#                     env,
#                     variants={
#                         # "obs_keys": [
#                         #     "hand_jnt",
#                         # ],
#                         "visual_keys": visual_keys,
#                     },
#                     variant_id=new_env_name,
#                 )
#                 env_names += [new_env_name]
#             except AssertionError as err:
#                 warnings.warn(
#                     f"Could not register {new_env_name}, the following error was raised: {err}"
#                 )
#         return env_names
#
