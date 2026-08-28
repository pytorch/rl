# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CraftGround (Minecraft) environment wrappers.

.. note:: **Minecraft ownership and licensing.** TorchRL does not ship, bundle
    or redistribute Minecraft. On first use, CraftGround's Gradle project
    downloads the Minecraft client and its assets from Mojang's servers onto
    the local machine and runs the game in offline mode. Offline mode removes
    the need for an account login during headless training; it does not remove
    the requirement to own the game: users are expected to own a valid
    Minecraft: Java Edition license, and any use of Minecraft is subject to the
    Minecraft End User License Agreement (https://www.minecraft.net/eula),
    including its restrictions on commercial exploitation. Never redistribute
    the downloaded game files (for instance inside public Docker images or CI
    caches). CraftGround is a separate, optional Python dependency that TorchRL
    does not vendor or redistribute. Its upstream repository currently ships a
    GPL-3.0 license file while its package metadata reports MIT; consult the
    upstream licensing information before distribution. See
    ``knowledge_base/MINECRAFT.md`` for details.
"""
from __future__ import annotations

import importlib.util
from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.libs.gym import (
    _gym_to_torchrl_spec_transform,
    GymWrapper,
    set_gym_backend,
)
from torchrl.envs.utils import _classproperty

if TYPE_CHECKING:
    from craftground.environment.environment import CraftGroundEnvironment
    from craftground.initial_environment_config import InitialEnvironmentConfig

_has_craftground = importlib.util.find_spec("craftground") is not None

# Boolean entries of CraftGround's v2 (MineRL-human-like) action space. The
# upstream gymnasium Dict space uses dotted keys ("hotbar.1") which cannot be
# used as TensorDict keys; we expose underscored variants and translate back in
# read_action.
_V2_BOOL_KEYS = (
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
) + tuple(f"hotbar_{i}" for i in range(1, 10))


class CraftGroundWrapper(GymWrapper):
    """CraftGround (Minecraft) environment wrapper.

    GitHub: https://github.com/yhs0602/CraftGround

    Documentation: https://yhs0602.github.io/CraftGround/

    Paper: Yun et al., "CraftGround: A Flexible Reinforcement Learning
    Environment Based on the Latest Minecraft" (2025).

    CraftGround runs a lightweight, headless-capable Minecraft client
    instrumented through a Fabric mod, and exposes it as a gymnasium
    environment. Observations are ego-centric RGB frames; actions follow
    either the MineDojo-style multi-discrete layout (v1) or a
    MineRL-human-like dictionary layout (v2).

    The wrapped environment is a *sandbox*: the underlying ``step`` always
    returns a zero reward and never terminates. Rewards and termination
    conditions are meant to be composed on top with TorchRL transforms
    (see the example below), or by sending Minecraft commands through
    ``env.add_command(...)`` and reading the resulting state.

    .. note:: **Minecraft ownership and licensing.** TorchRL does not
        distribute Minecraft. On the first :meth:`reset`, CraftGround's Gradle
        project downloads the Minecraft client from Mojang's servers onto the
        local machine and runs it in offline mode. Users are expected to own a
        valid Minecraft: Java Edition license; offline mode bypasses
        authentication, not ownership. Usage of Minecraft is governed by the
        Minecraft EULA (https://www.minecraft.net/eula), including its
        restrictions on commercial exploitation. Never redistribute the
        downloaded game files (e.g. in public Docker images or CI caches).
        CraftGround is a separate, optional dependency that TorchRL does not
        vendor or redistribute. Its upstream repository currently ships a
        GPL-3.0 license file while its package metadata reports MIT; consult the
        upstream licensing information before distribution. See
        ``knowledge_base/MINECRAFT.md`` in the TorchRL repository for details.

    .. note:: The environment is spawned lazily: constructing the wrapper only
        binds an IPC channel. The Minecraft client (a Java subprocess built and
        launched through Gradle) starts on the first :meth:`reset`, which can
        take several minutes on the very first run while Gradle downloads
        Minecraft and compiles the mod. Requires a JDK (OpenJDK 21) and, on
        headless machines, a virtual display (e.g. ``Xvfb``); see
        ``knowledge_base/MINECRAFT.md``.

    .. note:: Only the ``RAW`` (default) and ``PNG`` screen encoding modes are
        supported. ``RAW`` frames have shape ``(H, W, 3)``, ``PNG`` frames
        ``(3, W, H)``, both ``torch.uint8``. With a binocular configuration
        (``eye_distance > 0``) a second ``pixels_2`` entry is added.

    Args:
        env (craftground.environment.environment.CraftGroundEnvironment): the
            CraftGround environment instance to wrap.

    Keyword Args:
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent
            (:class:`torchrl.data.Categorical`), otherwise a one-hot encoding
            will be used (:class:`torchrl.data.OneHot`). Defaults to ``False``.
            Only used with the v1 (MineDojo-style) action space.
        device (torch.device, optional): if provided, the device on which the
            data is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): only ``torch.Size([])`` is
            supported, as CraftGround environments are not vectorized.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: an empty list; CraftGround environments are described
            by a ``craftground.InitialEnvironmentConfig`` rather than selected
            from a registry of task ids.

    Examples:
        >>> import craftground  # doctest: +SKIP
        >>> from torchrl.envs import TransformedEnv, StepCounter
        >>> from torchrl.envs.libs.craftground import CraftGroundWrapper
        >>> base = craftground.make(port=8023)  # doctest: +SKIP
        >>> env = CraftGroundWrapper(base)  # doctest: +SKIP
        >>> # the sandbox emits no reward: compose one with a transform,
        >>> # e.g. a step-count budget via StepCounter
        >>> env = TransformedEnv(env, StepCounter(max_steps=100))  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
        >>> assert td["next", "pixels"].dtype is torch.uint8  # doctest: +SKIP

    """

    git_url = "https://github.com/yhs0602/CraftGround"
    libname = "craftground"

    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib
        try:
            import craftground
        except ImportError as err:
            raise ImportError(
                "craftground not found. Install it with `pip install craftground` "
                "(requires OpenJDK 21 at runtime; see "
                "https://yhs0602.github.io/CraftGround/ and "
                "knowledge_base/MINECRAFT.md for setup and licensing notes)."
            ) from err
        cls._lib = craftground
        return craftground

    @_classproperty
    def available_envs(cls):
        # CraftGround has no registry of named tasks: environments are
        # parameterized by an InitialEnvironmentConfig instead.
        return []

    def _check_kwargs(self, kwargs: dict):
        super()._check_kwargs(kwargs)
        env = kwargs["env"]
        if not hasattr(env, "initial_env"):
            raise TypeError(
                "env must be a craftground CraftGroundEnvironment instance "
                "(missing the `initial_env` attribute). Use CraftGroundEnv to "
                "build one from a configuration."
            )

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ):
        if from_pixels:
            raise ValueError(
                "CraftGround environments are natively pixel-based: observations "
                "already contain a `pixels` entry, and `from_pixels` is not "
                "supported."
            )
        return super()._build_env(env, from_pixels=from_pixels, pixels_only=pixels_only)

    @staticmethod
    def _uses_v2_actions(env) -> bool:
        version = getattr(env, "action_space_version", None)
        return getattr(version, "name", None) == "V2_MINERL_HUMAN"

    def _make_specs(self, env, batch_size=None) -> None:
        config = env.initial_env
        height = config.imageSizeY
        width = config.imageSizeX
        mode = getattr(
            config.screen_encoding_mode, "name", str(config.screen_encoding_mode)
        )
        if mode == "RAW":
            pixels_shape = (height, width, 3)
        elif mode == "PNG":
            pixels_shape = (3, width, height)
        else:
            raise NotImplementedError(
                f"screen_encoding_mode {mode!r} is not supported by "
                "CraftGroundWrapper. Use ScreenEncodingMode.RAW (default) or "
                "ScreenEncodingMode.PNG."
            )
        pixels_spec = Bounded(
            low=0,
            high=255,
            shape=pixels_shape,
            dtype=torch.uint8,
            device=self.device,
        )
        observation_spec = Composite(pixels=pixels_spec, shape=self.batch_size)
        self._binocular = config.eye_distance > 0
        if self._binocular:
            observation_spec["pixels_2"] = pixels_spec.clone()

        self._v2_actions = self._uses_v2_actions(env)
        if self._v2_actions:
            action_spec = self._make_v2_action_spec()
        else:
            action_spec = _gym_to_torchrl_spec_transform(
                env.action_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )

        self.done_spec = self._make_done_spec()
        self.action_spec = action_spec
        self.reward_spec = Unbounded(shape=(1,), device=self.device)
        self.observation_spec = observation_spec

    _make_specs = set_gym_backend("gymnasium")(_make_specs)

    def _make_v2_action_spec(self) -> Composite:
        spec = {
            key: Categorical(2, shape=(), dtype=torch.bool, device=self.device)
            for key in _V2_BOOL_KEYS
        }
        spec["camera"] = Bounded(
            low=-180.0,
            high=180.0,
            shape=(2,),
            dtype=torch.float32,
            device=self.device,
        )
        return Composite(spec, shape=self.batch_size)

    def read_action(self, action):
        if self._v2_actions:
            # Translate the composite action back to CraftGround's dict format
            # (dotted hotbar keys, python bools, float32 camera).
            out = {}
            for key, value in action.items():
                if key == "camera":
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu()
                    out["camera"] = np.asarray(value, dtype=np.float32)
                else:
                    out[key.replace("hotbar_", "hotbar.")] = bool(value)
            return out
        return super().read_action(action)

    def _process_obs(self, observations):
        out = {"pixels": self._to_image(observations["pov"])}
        if self._binocular:
            out["pixels_2"] = self._to_image(observations["pov_2"])
        return out

    @staticmethod
    def _to_image(image):
        if isinstance(image, torch.Tensor):
            return image
        # CraftGround returns flipped, read-only numpy views; make them
        # contiguous and writable before tensor conversion.
        return np.ascontiguousarray(image)

    def _output_transform(self, step_outputs_tuple):
        # CraftGround's step returns (obs, reward, terminated, truncated, info)
        # where info is the observation dict itself and reward/terminated are
        # sandbox placeholders. The raw protobuf entry of the observation dict
        # is dropped here.
        observations, reward, terminated, truncated, _ = step_outputs_tuple
        observations = self._process_obs(observations)
        terminated = bool(terminated)
        truncated = bool(truncated)
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            None,
        )

    def _reset_output_transform(self, reset_outputs_tuple):
        observations, _ = reset_outputs_tuple
        return self._process_obs(observations), None


class CraftGroundEnv(CraftGroundWrapper):
    """CraftGround (Minecraft) environment built from a configuration.

    See :class:`CraftGroundWrapper` for behavior details and licensing notes.
    The constructor builds the environment through ``craftground.make(...)``.

    .. note:: **Minecraft ownership and licensing.** TorchRL does not
        distribute Minecraft. On the first :meth:`reset`, CraftGround's Gradle
        project downloads the Minecraft client from Mojang's servers onto the
        local machine and runs it in offline mode. Users are expected to own a
        valid Minecraft: Java Edition license; offline mode bypasses
        authentication, not ownership. Usage of Minecraft is governed by the
        Minecraft EULA (https://www.minecraft.net/eula), including its
        restrictions on commercial exploitation. Never redistribute the
        downloaded game files (e.g. in public Docker images or CI caches).
        CraftGround is a separate, optional dependency that TorchRL does not
        vendor or redistribute. Its upstream repository currently ships a
        GPL-3.0 license file while its package metadata reports MIT; consult the
        upstream licensing information before distribution. See
        ``knowledge_base/MINECRAFT.md`` in the TorchRL repository for details.

    Keyword Args:
        initial_env_config (craftground.InitialEnvironmentConfig, optional):
            the world/observation configuration (image size, game mode, world
            type, seed, initial commands, ...). Defaults to a fresh
            ``InitialEnvironmentConfig()``.
        mc_version (str, optional): the Minecraft version to run. Only
            ``"1.21"`` is currently functional upstream. Defaults to
            ``"1.21"``.
        port (int, optional): the IPC port used to communicate with the
            Minecraft process. A free port is picked automatically if the
            given one is busy. Defaults to ``8000``.
        action_space_version (craftground.ActionSpaceVersion, optional): the
            action layout, either ``V1_MINEDOJO`` (multi-discrete) or
            ``V2_MINERL_HUMAN`` (dict of booleans plus a continuous camera).
            Defaults to ``V1_MINEDOJO``.
        env_path (str, optional): path to a custom CraftGround Gradle project.
            Defaults to the project shipped with the installed
            ``craftground-runtime-mc121`` package.
        use_shared_memory (bool, optional): if ``True``, uses the shared-memory
            IPC backend instead of TCP sockets. Defaults to ``False``.
        verbose (bool, optional): enables CraftGround's verbose logging.
            Defaults to ``False``.
        craftground_kwargs (dict, optional): additional keyword arguments
            forwarded verbatim to ``craftground.make``.
        **kwargs: additional keyword arguments passed to
            :class:`CraftGroundWrapper` (e.g. ``device``).

    Examples:
        >>> from craftground.initial_environment_config import (  # doctest: +SKIP
        ...     InitialEnvironmentConfig, WorldType)
        >>> from torchrl.envs.libs.craftground import CraftGroundEnv
        >>> env = CraftGroundEnv(  # doctest: +SKIP
        ...     initial_env_config=InitialEnvironmentConfig(
        ...         image_width=114, image_height=64,
        ...         world_type=WorldType.SUPERFLAT,
        ...     ),
        ...     port=8023,
        ... )
        >>> td = env.reset()  # doctest: +SKIP
        >>> td["pixels"].shape  # doctest: +SKIP
        torch.Size([64, 114, 3])

    """

    def __init__(
        self,
        *,
        initial_env_config: InitialEnvironmentConfig | None = None,
        mc_version: Literal["1.21", "26.2"] = "1.21",
        port: int = 8000,
        action_space_version=None,
        env_path: str | None = None,
        use_shared_memory: bool = False,
        verbose: bool = False,
        craftground_kwargs: dict | None = None,
        **kwargs,
    ):
        craftground = self.lib
        if action_space_version is None:
            action_space_version = craftground.ActionSpaceVersion.V1_MINEDOJO
        craftground_kwargs = (
            dict(craftground_kwargs) if craftground_kwargs is not None else {}
        )
        env: CraftGroundEnvironment = craftground.make(
            initial_env_config=initial_env_config,
            mc_version=mc_version,
            port=port,
            action_space_version=action_space_version,
            env_path=env_path,
            use_shared_memory=use_shared_memory,
            verbose=verbose,
            **craftground_kwargs,
        )
        super().__init__(env=env, **kwargs)
