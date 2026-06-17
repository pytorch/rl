# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    NonTensor,
    Unbounded,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

_has_libero = importlib.util.find_spec("libero") is not None

__all__ = ["LiberoWrapper", "LiberoEnv"]


def _ensure_libero_config() -> None:
    """Pre-seed the LIBERO path config non-interactively.

    On first import, ``libero.libero`` prompts on stdin for a dataset
    location when no config file exists -- a deadlock in headless workers.
    This writes the default config (paths inside the installed package)
    beforehand, exactly as LIBERO itself would.
    """
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    import yaml

    # locate libero/libero without importing it (importing runs the prompt)
    spec = importlib.util.find_spec("libero.libero")
    root = os.path.dirname(spec.origin)
    paths = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.abspath(os.path.join(root, "..", "datasets")),
        "assets": os.path.join(root, "assets"),
    }
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as file:
        yaml.dump(paths, file)


def _get_suites() -> list[str]:
    if not _has_libero:
        raise ImportError("libero is not installed in your virtual environment.")
    _ensure_libero_config()
    from libero.libero import benchmark

    return sorted(benchmark.get_benchmark_dict())


class LiberoWrapper(_EnvWrapper):
    """LIBERO environment wrapper.

    GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO

    Paper: https://arxiv.org/abs/2306.03310 (LIBERO: Benchmarking Knowledge
    Transfer for Lifelong Robot Learning, Liu et al., 2023)

    LIBERO is a benchmark of language-conditioned tabletop manipulation
    tasks (robosuite/MuJoCo based) widely used to evaluate and fine-tune
    Vision-Language-Action policies (e.g. OpenVLA, SimpleVLA-RL). This
    wrapper exposes a LIBERO environment with the canonical VLA TensorDict
    schema (see :ref:`the VLA reference page <ref_vla>`): a ``uint8``
    ``[C, H, W]`` camera ``image`` (and optionally a ``wrist_image``) plus a
    proprioceptive ``state`` under ``observation``, the
    ``language_instruction`` string at the root, and a boolean ``success``
    entry. Episodes terminate on success and truncate after
    ``max_episode_steps``; the reward is the binary success indicator.

    Initial-state control is first-class: LIBERO ships a fixed set of
    initial simulator states per task (50 for the standard evaluation
    protocol), and the rollout-grouping mechanism GRPO-style group
    advantages require (n rollouts per initial state) is built in via
    ``group_repeats`` (see :class:`~torchrl.objectives.llm.MCAdvantage`).

    .. note::
        Rendering is offscreen (no display needed). On headless Linux set
        ``MUJOCO_GL=egl`` (or ``osmesa``); on macOS the default (``cgl``)
        works. One environment instance hosts one MuJoCo simulation: use
        :class:`~torchrl.envs.ParallelEnv` or an
        :class:`~torchrl.envs.AsyncEnvPool` with one instance per process
        for parallel collection. With EGL, :class:`~torchrl.envs.LiberoEnv`
        accepts a ``render_gpu_device_id`` argument to pin robosuite rendering
        to an EGL-visible GPU device.

    .. note::
        Importing ``libero`` for the first time normally prompts on stdin
        for a dataset path. This wrapper pre-seeds the default LIBERO config
        (honoring ``LIBERO_CONFIG_PATH``) so that headless workers never
        block on the prompt.

    Args:
        env (``libero.libero.envs.OffScreenRenderEnv``): the environment to
            wrap.

    Keyword Args:
        instruction (str, optional): the language instruction. Defaults to
            the instruction parsed from the task's BDDL file.
        init_states (np.ndarray, optional): a ``[N, D]`` array of initial
            simulator states to reset to (as returned by the LIBERO
            benchmark for each task). Required for ``init_state_mode`` /
            ``group_repeats``. Defaults to ``None`` (native random resets).
        init_state_mode (str, optional): how the initial state is selected
            at reset when ``init_states`` is provided. One of
            ``"random"`` (sample uniformly), ``"cycle"`` (walk through the
            init states in order -- the fixed-50-trials evaluation protocol)
            or ``"fixed"`` (always ``init_state_id``). Defaults to
            ``"random"``.
        init_state_id (int, optional): the initial state used with
            ``init_state_mode="fixed"``. Defaults to ``0``.
        group_repeats (int, optional): grouped-rollout mode: the same
            initial state is replayed for ``group_repeats`` consecutive
            episodes (one GRPO group) before the next one is selected, and
            an integer ``group_id`` observation entry identifies the group.
            Defaults to ``None``.
        group_id_offset (int, optional): added to the group ids. Group ids
            must be unique across concurrently collecting workers for group
            advantages to be computed correctly: give each worker a disjoint
            offset (e.g. ``worker_idx * 10**6``). Defaults to ``0``.
        group_id_mode (str, optional): how grouped-rollout ids are stamped.
            ``"episode"`` uses the local group counter
            (``episode_count // group_repeats``). ``"init_state"`` uses the
            selected initial-state id and requires ``init_state_mode`` to be
            ``"cycle"`` or ``"fixed"``. The latter is useful when several
            parallel workers share the same task and should form GRPO groups
            by initial state even if their episode lengths differ. Defaults to
            ``"episode"``.
        camera (str, optional): the camera rendered under
            ``("observation", "image")``. Defaults to ``"agentview"``.
        wrist_camera (str, optional): if provided (e.g.
            ``"robot0_eye_in_hand"``), this camera is exposed under
            ``("observation", "wrist_image")``. Defaults to ``None``.
        from_pixels (bool, optional): if ``True``, expose the rendered
            ``camera`` frame as a root ``pixels`` entry too (HWC ``uint8``),
            in addition to the CHW ``("observation", "image")`` the policy
            consumes. This is the torchrl pixels-rendering convention and
            feeds :class:`~torchrl.record.VideoRecorder` (default
            ``in_keys=["pixels"]``) with no extra render cost. Defaults to
            ``False``.
        flip_image (bool, optional): rotate the rendered images by 180
            degrees. LIBERO renders upside down relative to the standard
            human-view orientation used by the OpenVLA / SimpleVLA-RL
            pipelines. Defaults to ``True``.
        proprio_keys (tuple of str, optional): the low-dimensional observation
            entries concatenated (in order) into ``("observation", "state")``.
            Defaults to ``("robot0_eef_pos", "robot0_eef_quat",
            "robot0_gripper_qpos")``.
        max_episode_steps (int, optional): truncation horizon, in env steps.
            Defaults to ``512`` (the SimpleVLA-RL setting).
        settle_steps (int, optional): no-op steps (zero delta, gripper open)
            executed after setting the initial state so the simulation
            settles, as in the OpenVLA evaluation protocol. Defaults to
            ``10``.
        device (torch.device, optional): device of the output tensordicts.

    Attributes:
        available_envs (list of str): the available task-suite names.

    Examples:
        >>> import os
        >>> from libero.libero import benchmark, get_libero_path
        >>> from libero.libero.envs import OffScreenRenderEnv
        >>> from torchrl.envs import LiberoWrapper
        >>> suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        >>> task = suite.get_task(0)
        >>> env = OffScreenRenderEnv(
        ...     bddl_file_name=os.path.join(
        ...         get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ...     ),
        ...     camera_heights=256,
        ...     camera_widths=256,
        ... )
        >>> env = LiberoWrapper(env, instruction=task.language)
        >>> td = env.reset()
        >>> td["observation", "image"].shape
        torch.Size([3, 256, 256])
    """

    git_url = "https://github.com/Lifelong-Robot-Learning/LIBERO"
    libname = "libero"

    @property
    def lib(self):
        import libero

        return libero

    @_classproperty
    def available_envs(cls):
        if not _has_libero:
            return []
        return _get_suites()

    def __init__(
        self,
        env=None,  # noqa: F821
        *,
        instruction: str | None = None,
        init_states: np.ndarray | None = None,
        init_state_mode: Literal["random", "cycle", "fixed"] = "random",
        init_state_id: int = 0,
        group_repeats: int | None = None,
        group_id_offset: int = 0,
        group_id_mode: Literal["episode", "init_state"] = "episode",
        camera: str = "agentview",
        wrist_camera: str | None = None,
        from_pixels: bool = False,
        flip_image: bool = True,
        proprio_keys: tuple[str, ...] = (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        max_episode_steps: int | None = 512,
        settle_steps: int = 10,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
        if init_state_mode not in ("random", "cycle", "fixed"):
            raise ValueError(
                "init_state_mode must be one of 'random', 'cycle' or 'fixed', "
                f"got {init_state_mode!r}."
            )
        if group_repeats is not None and group_repeats < 1:
            raise ValueError(f"group_repeats must be >= 1, got {group_repeats}.")
        if group_id_mode not in ("episode", "init_state"):
            raise ValueError(
                "group_id_mode must be one of 'episode' or 'init_state', "
                f"got {group_id_mode!r}."
            )
        if group_id_mode == "init_state" and init_state_mode == "random":
            raise ValueError(
                "group_id_mode='init_state' requires init_state_mode='cycle' "
                "or 'fixed'."
            )
        self.instruction = instruction
        self.init_states = init_states
        self.init_state_mode = init_state_mode
        self.init_state_id = int(init_state_id)
        self.group_repeats = group_repeats
        self.group_id_offset = int(group_id_offset)
        self.group_id_mode = group_id_mode
        self.camera = camera
        self.wrist_camera = wrist_camera
        self.from_pixels = bool(from_pixels)
        self.flip_image = bool(flip_image)
        self.proprio_keys = tuple(proprio_keys)
        self.max_episode_steps = max_episode_steps
        self.settle_steps = int(settle_steps)
        self._episode_count = 0
        self._elapsed = 0
        self._rng = torch.Generator()
        super().__init__(**kwargs)

    def _build_env(self, env):  # noqa: F821
        if self.instruction is None:
            self.instruction = str(env.language_instruction)
        return env

    def _check_kwargs(self, kwargs: dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")

    @property
    def _dummy_action(self) -> np.ndarray:
        # zero delta-pose, gripper open: the OpenVLA settle action
        action = np.zeros(self._action_dim, dtype=np.float64)
        action[-1] = -1.0
        return action

    def _make_specs(self, env) -> None:  # noqa: F821
        obs = env.reset()
        low, high = env.env.action_spec
        self._action_dim = len(low)
        image = self._read_image(obs[f"{self.camera}_image"])
        state = self._read_state(obs)
        observation = Composite(
            image=Unbounded(shape=image.shape, dtype=torch.uint8, device=self.device),
            state=Unbounded(shape=state.shape, dtype=torch.float32, device=self.device),
            shape=(),
        )
        if self.wrist_camera is not None:
            wrist = self._read_image(obs[f"{self.wrist_camera}_image"])
            observation["wrist_image"] = Unbounded(
                shape=wrist.shape, dtype=torch.uint8, device=self.device
            )
        self.observation_spec = Composite(
            observation=observation,
            language_instruction=NonTensor(
                shape=(), example_data=self.instruction, device=self.device
            ),
            success=Categorical(2, dtype=torch.bool, shape=(1,), device=self.device),
            shape=(),
        )
        if self.group_repeats is not None:
            self.observation_spec["group_id"] = Unbounded(
                shape=(1,), dtype=torch.int64, device=self.device
            )
        if self.from_pixels:
            # the camera is always rendered for the policy image; expose it as
            # a root HWC ``pixels`` entry too (the torchrl from_pixels / video
            # convention). image is CHW (C, H, W) -> pixels is (H, W, C).
            self.observation_spec["pixels"] = Unbounded(
                shape=(image.shape[1], image.shape[2], image.shape[0]),
                dtype=torch.uint8,
                device=self.device,
            )
        self.action_spec = Bounded(
            low=torch.as_tensor(low, dtype=torch.float32),
            high=torch.as_tensor(high, dtype=torch.float32),
            shape=(self._action_dim,),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_spec = Bounded(low=0.0, high=1.0, shape=(1,), device=self.device)
        self.done_spec = Composite(
            done=Categorical(2, dtype=torch.bool, shape=(1,), device=self.device),
            terminated=Categorical(2, dtype=torch.bool, shape=(1,), device=self.device),
            truncated=Categorical(2, dtype=torch.bool, shape=(1,), device=self.device),
            shape=(),
        )

    def _init_env(self) -> None:
        # runs after _build_env, which may have auto-loaded the init states
        if self.init_states is None and (
            self.group_repeats is not None or self.init_state_mode != "random"
        ):
            raise ValueError(
                "init-state control (group_repeats / init_state_mode / "
                "init_state_id) requires init_states."
            )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._env.seed(seed)
            self._rng.manual_seed(seed)

    def _select_init_state(self) -> np.ndarray | None:
        if self.init_states is None:
            self._current_init_state_index = None
            return None
        n_states = len(self.init_states)
        repeats = self.group_repeats if self.group_repeats is not None else 1
        group = self._episode_count // repeats
        if self.init_state_mode == "fixed":
            index = self.init_state_id
        elif self.init_state_mode == "cycle":
            index = group % n_states
        else:  # "random": one fresh draw per group, replayed within the group
            if self._episode_count % repeats == 0 or self._group_init_index is None:
                self._group_init_index = int(
                    torch.randint(n_states, (), generator=self._rng)
                )
            index = self._group_init_index
        self._current_init_state_index = int(index)
        return self.init_states[index]

    _group_init_index: int | None = None
    _current_init_state_index: int | None = None

    def _read_image(self, image: np.ndarray) -> torch.Tensor:
        if self.flip_image:
            image = image[::-1, ::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()
        )

    def _read_state(self, obs: dict) -> torch.Tensor:
        parts = [np.atleast_1d(np.asarray(obs[key])) for key in self.proprio_keys]
        return torch.from_numpy(np.concatenate(parts)).to(torch.float32)

    def _read_obs(self, obs: dict, *, success: bool) -> TensorDict:
        image = self._read_image(obs[f"{self.camera}_image"])
        out = TensorDict(
            {
                "observation": {
                    "image": image,
                    "state": self._read_state(obs),
                },
                "language_instruction": self.instruction,
                "success": torch.tensor([success], dtype=torch.bool),
            },
            batch_size=[],
            device=self.device,
        )
        if self.from_pixels:
            # HWC view of the same rendered frame (no extra render)
            out["pixels"] = image.permute(1, 2, 0).contiguous()
        if self.wrist_camera is not None:
            out["observation", "wrist_image"] = self._read_image(
                obs[f"{self.wrist_camera}_image"]
            )
        if self.group_repeats is not None:
            out["group_id"] = torch.tensor([self._current_group], dtype=torch.int64)
        return out

    _current_group: int = 0

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        obs = self._env.reset()
        init_state = self._select_init_state()
        if init_state is not None:
            obs = self._env.set_init_state(init_state)
        for _ in range(self.settle_steps):
            obs, _, _, _ = self._env.step(self._dummy_action)
        self._elapsed = 0
        if self.group_repeats is not None:
            # stamp the group for the whole upcoming episode (the counter
            # advances below, so steps keep reporting this episode's group)
            if self.group_id_mode == "init_state":
                if self._current_init_state_index is None:
                    raise RuntimeError(
                        "group_id_mode='init_state' requires an active init state."
                    )
                group = self._current_init_state_index
            else:
                group = self._episode_count // self.group_repeats
            self._current_group = self.group_id_offset + group
        out = self._read_obs(obs, success=bool(self._env.check_success()))
        self._episode_count += 1
        out.update(self.full_done_spec.zero())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action").detach().cpu().numpy().astype(np.float64)
        obs, _, _, _ = self._env.step(action)
        self._elapsed += 1
        success = bool(self._env.check_success())
        truncated = (
            self.max_episode_steps is not None
            and self._elapsed >= self.max_episode_steps
        )
        out = self._read_obs(obs, success=success)
        out["reward"] = torch.tensor([float(success)])
        out["terminated"] = torch.tensor([success], dtype=torch.bool)
        out["truncated"] = torch.tensor([truncated and not success], dtype=torch.bool)
        out["done"] = torch.tensor([success or truncated], dtype=torch.bool)
        return out

    def close(self, *, raise_if_closed: bool = True) -> None:
        if not self.is_closed:
            self._env.close()
        super().close(raise_if_closed=raise_if_closed)


class LiberoEnv(LiberoWrapper):
    """LIBERO environment built from a task-suite name and task id.

    GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO

    Paper: https://arxiv.org/abs/2306.03310 (LIBERO: Benchmarking Knowledge
    Transfer for Lifelong Robot Learning, Liu et al., 2023)

    See :class:`~torchrl.envs.LiberoWrapper` for the full TensorDict schema
    and keyword arguments. This constructor builds the underlying
    ``OffScreenRenderEnv`` from the benchmark registry, fetches the task's
    language instruction and its fixed initial states (50 per task in the
    standard suites) and wires them into the init-state control machinery.

    Args:
        task_suite (str): the task-suite name, from :attr:`~.available_envs`
            (e.g. ``"libero_spatial"``, ``"libero_object"``,
            ``"libero_goal"``, ``"libero_10"``).
        task_id (int): the task index inside the suite.

    Keyword Args:
        camera_height (int, optional): rendered image height. Defaults to
            ``256`` (the SimpleVLA-RL / OpenVLA-OFT resolution).
        camera_width (int, optional): rendered image width. Defaults to
            ``256``.
        render_gpu_device_id (int, optional): GPU device id used by
            robosuite for offscreen rendering. This is the EGL-visible device
            id inside the process/container, not necessarily the global CUDA
            ordinal. Use this to spread LIBERO render workers across multiple
            GPUs, e.g. ``worker_idx % num_render_gpus``. If omitted, robosuite
            selects its default device. Defaults to ``None``.
        env_kwargs (dict, optional): extra keyword arguments forwarded to
            ``OffScreenRenderEnv`` (e.g. ``horizon``).
        **kwargs: see :class:`~torchrl.envs.LiberoWrapper`.

    Examples:
        >>> from torchrl.envs import LiberoEnv
        >>> env = LiberoEnv(
        ...     "libero_spatial",
        ...     task_id=0,
        ...     camera_height=128,
        ...     camera_width=128,
        ...     init_state_mode="cycle",
        ... )
        >>> td = env.reset()
        >>> td["language_instruction"]
        'pick up the black bowl between the plate and the ramekin and place it on the plate'
        >>> td["observation", "image"].shape
        torch.Size([3, 128, 128])
    """

    def __init__(
        self,
        task_suite: str | None = None,
        task_id: int | None = None,
        *,
        camera_height: int = 256,
        camera_width: int = 256,
        render_gpu_device_id: int | None = None,
        env_kwargs: dict | None = None,
        **kwargs,
    ):
        if task_suite is None or task_id is None:
            raise TypeError("task_suite and task_id must both be specified.")
        kwargs["task_suite"] = task_suite
        kwargs["task_id"] = int(task_id)
        kwargs["camera_height"] = int(camera_height)
        kwargs["camera_width"] = int(camera_width)
        env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
        if render_gpu_device_id is not None:
            if "render_gpu_device_id" in env_kwargs and int(
                env_kwargs["render_gpu_device_id"]
            ) != int(render_gpu_device_id):
                raise ValueError(
                    "render_gpu_device_id was provided both directly and in "
                    "env_kwargs with different values."
                )
            env_kwargs["render_gpu_device_id"] = int(render_gpu_device_id)
        kwargs["env_kwargs"] = env_kwargs
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        if "task_suite" not in kwargs or "task_id" not in kwargs:
            raise TypeError("task_suite and task_id must both be specified.")

    @staticmethod
    def _load_init_states(suite, task) -> np.ndarray:
        # suite.get_task_init_states relies on torch.load defaults that
        # reject these (trusted, benchmark-shipped) pickled arrays on
        # torch>=2.6: load the file directly instead
        from libero.libero import get_libero_path

        path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        return torch.load(path, weights_only=False)

    def _build_env(self, task_suite: str, task_id: int, camera_height, camera_width, env_kwargs):  # type: ignore[override]
        if not _has_libero:
            raise ImportError(
                "libero is not installed in your virtual environment. Install it "
                "from source: git clone https://github.com/Lifelong-Robot-Learning/LIBERO "
                "&& pip install -e LIBERO (plus robosuite and bddl)."
            )
        _ensure_libero_config()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite not in benchmark_dict:
            raise ValueError(
                f"Unknown task suite {task_suite!r}. Available suites: "
                f"{sorted(benchmark_dict)}."
            )
        suite = benchmark_dict[task_suite]()
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(
                f"task_id {task_id} out of range for {task_suite!r} "
                f"({suite.n_tasks} tasks)."
            )
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            ),
            camera_heights=camera_height,
            camera_widths=camera_width,
            **env_kwargs,
        )
        if self.instruction is None:
            self.instruction = str(task.language)
        if self.init_states is None:
            self.init_states = self._load_init_states(suite, task)
        self.task_suite = task_suite
        self.task_id = task_id
        return env
