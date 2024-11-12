# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from tensordict import TensorDict, TensorDictBase

from torchrl.envs.common import _EnvPostInit
from torchrl.envs.utils import _classproperty

_has_jumanji = importlib.util.find_spec("jumanji") is not None

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    DEVICE_TYPING,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.gym_like import GymLikeEnv

from torchrl.envs.libs.jax_utils import (
    _extract_spec,
    _ndarray_to_tensor,
    _object_to_tensordict,
    _tensordict_to_object,
    _tree_flatten,
    _tree_reshape,
)


def _get_envs():
    if not _has_jumanji:
        raise ImportError("Jumanji is not installed in your virtual environment.")
    import jumanji

    return jumanji.registered_environments()


def _jumanji_to_torchrl_spec_transform(
    spec,
    dtype: Optional[torch.dtype] = None,
    device: DEVICE_TYPING = None,
    categorical_action_encoding: bool = True,
) -> TensorSpec:
    import jumanji

    if isinstance(spec, jumanji.specs.DiscreteArray):
        action_space_cls = Categorical if categorical_action_encoding else OneHot
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return action_space_cls(spec.num_values, dtype=dtype, device=device)
    if isinstance(spec, jumanji.specs.MultiDiscreteArray):
        action_space_cls = (
            MultiCategorical if categorical_action_encoding else MultiOneHot
        )
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return action_space_cls(
            torch.as_tensor(np.asarray(spec.num_values)), dtype=dtype, device=device
        )
    elif isinstance(spec, jumanji.specs.BoundedArray):
        shape = spec.shape
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return Bounded(
            shape=shape,
            low=np.asarray(spec.minimum),
            high=np.asarray(spec.maximum),
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, jumanji.specs.Array):
        shape = spec.shape
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        if dtype in (torch.float, torch.double, torch.half):
            return Unbounded(shape=shape, dtype=dtype, device=device)
        else:
            return Unbounded(shape=shape, dtype=dtype, device=device)
    elif isinstance(spec, jumanji.specs.Spec) and hasattr(spec, "__dict__"):
        new_spec = {}
        for key, value in spec.__dict__.items():
            if isinstance(value, jumanji.specs.Spec):
                if key.endswith("_obs"):
                    key = key[:-4]
                if key.endswith("_spec"):
                    key = key[:-5]
                new_spec[key] = _jumanji_to_torchrl_spec_transform(
                    value, dtype, device, categorical_action_encoding
                )
        return Composite(**new_spec)
    else:
        raise TypeError(f"Unsupported spec type {type(spec)}")


class _JumanjiMakeRender(_EnvPostInit):
    def __call__(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if instance.from_pixels:
            return instance.make_render()
        return instance


class JumanjiWrapper(GymLikeEnv, metaclass=_JumanjiMakeRender):
    """Jumanji's environment wrapper.

    Jumanji offers a vectorized simulation framework based on Jax.
    TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
    but computational graphs can still be built on top of the simulated trajectories,
    allowing for backpropagation through the rollout.

    GitHub: https://github.com/instadeepai/jumanji

    Doc: https://instadeepai.github.io/jumanji/

    Paper: https://arxiv.org/abs/2306.09884

    .. note:: For better performance, turn `jit` on when instantiating this class.
        The `jit` attribute can also be flipped during code execution:

            >>> env.jit = True # Used jit
            >>> env.jit = False # eager

    Args:
        env (jumanji.env.Environment): the env to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.

    Keyword Args:
        batch_size (torch.Size, optional): the batch size of the environment.
            With ``jumanji``, this indicates the number of vectorized environments.
            If the batch-size is empty, the environment is not batch-locked and an arbitrary number
            of environments can be executed simultaneously.
            Defaults to ``torch.Size([])``.

                >>> import jumanji
                >>> from torchrl.envs import JumanjiWrapper
                >>> base_env = jumanji.make("Snake-v1")
                >>> env = JumanjiWrapper(base_env)
                >>> # Set the batch-size of the TensorDict instead of the env allows to control the number
                >>> #  of envs being run simultaneously
                >>> tdreset = env.reset(TensorDict(batch_size=[32]))
                >>> # Execute a rollout until all envs are done or max steps is reached, whichever comes first
                >>> rollout = env.rollout(100, break_when_all_done=True, auto_reset=False, tensordict=tdreset)

        from_pixels (bool, optional): Whether the environment should render its output.
            This will drastically impact the environment throughput. Only the first environment
            will be rendered. See :meth:`~torchrl.envs.JumanjiWrapper.render` for more information.
            Defaults to `False`.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.
        jit (bool, optional): whether the step and reset method should be wrapped in `jit`.
            Defaults to ``False``.

    Attributes:
        available_envs: environments availalbe to build

    Examples:
        >>> import jumanji
        >>> from torchrl.envs import JumanjiWrapper
        >>> base_env = jumanji.make("Snake-v1")
        >>> env = JumanjiWrapper(base_env)
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                                body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                                body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
                                fruit_position: TensorDict(
                                    fields={
                                        col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                        row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=cpu,
                                    is_shared=False),
                                head_position: TensorDict(
                                    fields={
                                        col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                        row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=cpu,
                                    is_shared=False),
                                key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
                                length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                state: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                        body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
                        fruit_position: TensorDict(
                            fields={
                                col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        head_position: TensorDict(
                            fields={
                                col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
                        length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['Game2048-v1',
         'Maze-v0',
         'Cleaner-v0',
         'CVRP-v1',
         'MultiCVRP-v0',
         'Minesweeper-v0',
         'RubiksCube-v0',
         'Knapsack-v1',
         'Sudoku-v0',
         'Snake-v1',
         'TSP-v1',
         'Connector-v2',
         'MMST-v0',
         'GraphColoring-v0',
         'RubiksCube-partly-scrambled-v0',
         'RobotWarehouse-v0',
         'Tetris-v0',
         'BinPack-v2',
         'Sudoku-very-easy-v0',
         'JobShop-v0']

    To take advante of Jumanji, one usually executes multiple environments at the
    same time.

        >>> import jumanji
        >>> from torchrl.envs import JumanjiWrapper
        >>> base_env = jumanji.make("Snake-v1")
        >>> env = JumanjiWrapper(base_env, batch_size=[10])
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)

    In the following example, we iteratively test different batch sizes
    and report the execution time for a short rollout:

    Examples:
        >>> from torch.utils.benchmark import Timer
        >>> for batch_size in [4, 16, 128]:
        ...     timer = Timer(
        ...     '''
        ... env.rollout(100)
        ... ''',
        ... setup=f'''
        ... from torchrl.envs import JumanjiWrapper
        ... import jumanji
        ... env = JumanjiWrapper(jumanji.make('Snake-v1'), batch_size=[{batch_size}])
        ... env.set_seed(0)
        ... env.rollout(2)
        ... ''')
        ...     print(batch_size, timer.timeit(number=10))
        4
        env.rollout(100)
        setup: [...]
        Median: 122.40 ms
        2 measurements, 1 runs per measurement, 1 thread

        16
        env.rollout(100)
        setup: [...]
        Median: 134.39 ms
        2 measurements, 1 runs per measurement, 1 thread

        128
        env.rollout(100)
        setup: [...]
        Median: 172.31 ms
        2 measurements, 1 runs per measurement, 1 thread

    """

    git_url = "https://github.com/instadeepai/jumanji"
    libname = "jumanji"

    @_classproperty
    def available_envs(cls):
        if not _has_jumanji:
            return []
        return sorted(_get_envs())

    @property
    def lib(self):
        import jumanji

        if version.parse(jumanji.__version__) < version.parse("1.0.0"):
            raise ImportError("jumanji version must be >= 1.0.0")
        return jumanji

    def __init__(
        self,
        env: "jumanji.env.Environment" = None,  # noqa: F821
        categorical_action_encoding=True,
        jit: bool = True,
        **kwargs,
    ):
        if not _has_jumanji:
            raise ImportError(
                "jumanji is not installed or importing it failed. Consider checking your installation."
            )
        self.categorical_action_encoding = categorical_action_encoding
        if env is not None:
            kwargs["env"] = env
        batch_locked = kwargs.pop("batch_locked", kwargs.get("batch_size") is not None)
        super().__init__(**kwargs)
        self._batch_locked = batch_locked
        self.jit = jit

    @property
    def jit(self):
        return self._jit

    @jit.setter
    def jit(self, value):
        self._jit = value
        if value:
            import jax

            self._env_reset = jax.jit(self._env.reset)
            self._env_step = jax.jit(self._env.step)
        else:
            self._env_reset = self._env.reset
            self._env_step = self._env.step

    def _build_env(
        self,
        env,
        _seed: Optional[int] = None,
        from_pixels: bool = False,
        render_kwargs: Optional[dict] = None,
        pixels_only: bool = False,
        camera_id: Union[int, str] = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        return env

    def make_render(self):
        """Returns a transformed environment that can be rendered.

        Examples:
            >>> from torchrl.envs import JumanjiEnv
            >>> from torchrl.record import CSVLogger, VideoRecorder
            >>>
            >>> envname = JumanjiEnv.available_envs[-1]
            >>> logger = CSVLogger("jumanji", video_format="mp4", video_fps=2)
            >>> env = JumanjiEnv(envname, from_pixels=True)
            >>>
            >>> env = env.append_transform(
            ...     VideoRecorder(logger=logger, in_keys=["pixels"], tag=envname)
            ... )
            >>> env.set_seed(0)
            >>> r = env.rollout(100)
            >>> env.transform.dump()

        """
        from torchrl.record import PixelRenderTransform

        return self.append_transform(
            PixelRenderTransform(
                out_keys=["pixels"],
                pass_tensordict=True,
                as_non_tensor=bool(self.batch_size),
                as_numpy=bool(self.batch_size),
            )
        )

    def _make_state_example(self, env):
        import jax
        from jax import numpy as jnp

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state, _ = jax.vmap(env.reset)(jnp.stack(keys))
        state = _tree_reshape(state, self.batch_size)
        return state

    def _make_state_spec(self, env) -> TensorSpec:
        import jax

        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict)
        return state_spec

    def _make_action_spec(self, env) -> TensorSpec:
        action_spec = _jumanji_to_torchrl_spec_transform(
            env.action_spec,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )
        action_spec = action_spec.expand(*self.batch_size, *action_spec.shape)
        return action_spec

    def _make_observation_spec(self, env) -> TensorSpec:
        jumanji = self.lib

        spec = env.observation_spec
        new_spec = _jumanji_to_torchrl_spec_transform(spec, device=self.device)
        if isinstance(spec, jumanji.specs.Array):
            return Composite(observation=new_spec).expand(self.batch_size)
        elif isinstance(spec, jumanji.specs.Spec):
            return Composite(**{k: v for k, v in new_spec.items()}).expand(
                self.batch_size
            )
        else:
            raise TypeError(f"Unsupported spec type {type(spec)}")

    def _make_reward_spec(self, env) -> TensorSpec:
        reward_spec = _jumanji_to_torchrl_spec_transform(
            env.reward_spec, device=self.device
        )
        if not len(reward_spec.shape):
            reward_spec.shape = torch.Size([1])
        return reward_spec.expand([*self.batch_size, *reward_spec.shape])

    def _make_specs(self, env: "jumanji.env.Environment") -> None:  # noqa: F821

        # extract spec from jumanji definition
        self.action_spec = self._make_action_spec(env)
        self.observation_spec = self._make_observation_spec(env)
        self.reward_spec = self._make_reward_spec(env)

        # extract state spec from instance
        state_spec = self._make_state_spec(env).expand(self.batch_size)
        self.state_spec["state"] = state_spec
        self.observation_spec["state"] = state_spec.clone()

        # build state example for data conversion
        self._state_example = self._make_state_example(env)

    def _check_kwargs(self, kwargs: Dict):
        jumanji = self.lib
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (jumanji.env.Environment,)):
            raise TypeError("env is not of type 'jumanji.env.Environment'.")

    def _init_env(self):
        pass

    @property
    def key(self):
        key = getattr(self, "_key", None)
        if key is None:
            raise RuntimeError(
                "the env.key attribute wasn't found. Make sure to call `env.set_seed(seed)` before any interaction."
            )
        return key

    @key.setter
    def key(self, value):
        self._key = value

    def _set_seed(self, seed):
        import jax

        if seed is None:
            raise Exception("Jumanji requires an integer seed.")
        self.key = jax.random.PRNGKey(seed)

    def read_state(self, state, batch_size=None):
        state_dict = _object_to_tensordict(
            state, self.device, self.batch_size if batch_size is None else batch_size
        )
        return self.state_spec["state"].encode(state_dict)

    def read_obs(self, obs, batch_size=None):
        from jax import numpy as jnp

        if isinstance(obs, (list, jnp.ndarray, np.ndarray)):
            obs_dict = _ndarray_to_tensor(obs).to(self.device)
        else:
            obs_dict = _object_to_tensordict(
                obs, self.device, self.batch_size if batch_size is None else batch_size
            )
        return super().read_obs(obs_dict)

    def render(
        self,
        tensordict,
        matplotlib_backend: str | None = None,
        as_numpy: bool = False,
        **kwargs,
    ):
        """Renders the environment output given an input tensordict.

        This method is intended to be called by the :class:`~torchrl.record.PixelRenderTransform`
        created whenever `from_pixels=True` is selected.
        To create an appropriate rendering transform, use a similar call as bellow:

            >>> from torchrl.record import PixelRenderTransform
            >>> matplotlib_backend = None # Change this value if a specific matplotlib backend has to be used.
            >>> env = env.append_transform(
            ...     PixelRenderTransform(out_keys=["pixels"], pass_tensordict=True, matplotlib_backend=matplotlib_backend)
            ... )

        This pipeline will write a `"pixels"` entry in your output tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict containing a state to represent
            matplotlib_backend (str, optional): the matplotlib backend
            as_numpy (bool, optional): if ``False``, the np.ndarray will be converted to a torch.Tensor.
                Defaults to ``False``.

        """
        import io

        import jax
        import jax.numpy as jnp
        import jumanji

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import PIL
            import torchvision.transforms.v2.functional
        except ImportError as err:
            raise ImportError(
                "Rendering with Jumanji requires torchvision, matplotlib and PIL to be installed."
            ) from err

        if matplotlib_backend is not None:
            matplotlib.use(matplotlib_backend)

        # Get only one env
        _state_example = self._state_example
        while tensordict.ndim:
            tensordict = tensordict[0]
            _state_example = jax.tree_util.tree_map(
                lambda x: jnp.take(x, 0, axis=0), _state_example
            )
        # Patch jumanji is_notebook
        is_notebook = jumanji.environments.is_notebook
        try:
            jumanji.environments.is_notebook = lambda: False

            isinteractive = plt.isinteractive()
            plt.ion()
            buf = io.BytesIO()
            state = _tensordict_to_object(
                tensordict.get("state"),
                _state_example,
                batch_size=tensordict.batch_size if not self.batch_locked else None,
            )
            self._env.render(state, **kwargs)
            plt.savefig(buf, format="png")
            buf.seek(0)
            # Load the image into a PIL object.
            img = PIL.Image.open(buf)
            img_array = torchvision.transforms.v2.functional.pil_to_tensor(img)
            if not isinteractive:
                plt.ioff()
            plt.close()
            if not as_numpy:
                return img_array[:3]
            return img_array[:3].numpy().copy()
        finally:
            jumanji.environments.is_notebook = is_notebook

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        import jax

        if self.batch_locked:
            batch_size = self.batch_size
        else:
            batch_size = tensordict.batch_size

        # prepare inputs
        state = _tensordict_to_object(
            tensordict.get("state"),
            self._state_example,
            batch_size=tensordict.batch_size if not self.batch_locked else None,
        )
        action = self.read_action(tensordict.get("action"))

        # flatten batch size into vector
        state = _tree_flatten(state, batch_size)
        action = _tree_flatten(action, batch_size)

        # jax vectorizing map on env.step
        state, timestep = jax.vmap(self._env_step)(state, action)

        # reshape batch size from vector
        state = _tree_reshape(state, batch_size)
        timestep = _tree_reshape(timestep, batch_size)

        # collect outputs
        state_dict = self.read_state(state, batch_size=batch_size)
        obs_dict = self.read_obs(timestep.observation, batch_size=batch_size)
        reward = self.read_reward(np.asarray(timestep.reward))
        done = timestep.step_type == self.lib.types.StepType.LAST
        done = _ndarray_to_tensor(done).view(torch.bool).to(self.device)

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        tensordict_out.set("terminated", done)
        # tensordict_out.set("terminated", done)
        tensordict_out["state"] = state_dict

        return tensordict_out

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        import jax
        from jax import numpy as jnp

        if self.batch_locked or tensordict is None:
            numel = self.numel()
            batch_size = self.batch_size
        elif tensordict is not None:
            numel = tensordict.numel()
            batch_size = tensordict.batch_size

        # generate random keys
        self.key, *keys = jax.random.split(self.key, numel + 1)

        # jax vectorizing map on env.reset
        state, timestep = jax.vmap(self._env_reset)(jnp.stack(keys))

        # reshape batch size from vector
        state = _tree_reshape(state, batch_size)
        timestep = _tree_reshape(timestep, batch_size)

        # collect outputs
        state_dict = self.read_state(state, batch_size=batch_size)
        obs_dict = self.read_obs(timestep.observation, batch_size=batch_size)
        if not self.batch_locked:
            done_td = self.full_done_spec.zero(batch_size)
        else:
            done_td = self.full_done_spec.zero()

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=batch_size,
            device=self.device,
        )
        tensordict_out.update(done_td)
        tensordict_out["state"] = state_dict

        return tensordict_out

    def read_reward(self, reward):
        """Reads the reward and maps it to the reward space.

        Args:
            reward (torch.Tensor or TensorDict): reward to be mapped.

        """
        if isinstance(reward, int) and reward == 0:
            return self.reward_spec.zero()
        if self.batch_locked:
            reward = self.reward_spec.encode(reward, ignore_device=True)
        else:
            reward = torch.as_tensor(reward)
            if not reward.ndim or (reward.shape[-1] != self.reward_spec.shape[-1]):
                reward = reward.unsqueeze(-1)

        if reward is None:
            reward = torch.tensor(np.nan).expand(self.reward_spec.shape)

        return reward

    def _output_transform(self, step_outputs_tuple: Tuple) -> Tuple:
        ...

    def _reset_output_transform(self, reset_outputs_tuple: Tuple) -> Tuple:
        ...


class JumanjiEnv(JumanjiWrapper):
    """Jumanji environment wrapper built with the environment name.

    Jumanji offers a vectorized simulation framework based on Jax.
    TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
    but computational graphs can still be built on top of the simulated trajectories,
    allowing for backpropagation through the rollout.

    GitHub: https://github.com/instadeepai/jumanji

    Doc: https://instadeepai.github.io/jumanji/

    Paper: https://arxiv.org/abs/2306.09884

    Args:
        env_name (str): the name of the environment to wrap. Must be part of :attr:`~.available_envs`.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.

    Keyword Args:
        from_pixels (bool, optional): Not yet supported.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            With ``jumanji``, this indicates the number of vectorized environments.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments availalbe to build

    Examples:
        >>> from torchrl.envs import JumanjiEnv
        >>> env = JumanjiEnv("Snake-v1")
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        grid: Tensor(shape=torch.Size([12, 12, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                                body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                                body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
                                fruit_position: TensorDict(
                                    fields={
                                        col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                        row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=cpu,
                                    is_shared=False),
                                head_position: TensorDict(
                                    fields={
                                        col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                        row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=cpu,
                                    is_shared=False),
                                key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
                                length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                state: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        body: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                        body_state: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.int32, is_shared=False),
                        fruit_position: TensorDict(
                            fields={
                                col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        head_position: TensorDict(
                            fields={
                                col: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                                row: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        key: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int32, is_shared=False),
                        length: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        tail: Tensor(shape=torch.Size([12, 12]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                step_count: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['Game2048-v1',
         'Maze-v0',
         'Cleaner-v0',
         'CVRP-v1',
         'MultiCVRP-v0',
         'Minesweeper-v0',
         'RubiksCube-v0',
         'Knapsack-v1',
         'Sudoku-v0',
         'Snake-v1',
         'TSP-v1',
         'Connector-v2',
         'MMST-v0',
         'GraphColoring-v0',
         'RubiksCube-partly-scrambled-v0',
         'RobotWarehouse-v0',
         'Tetris-v0',
         'BinPack-v2',
         'Sudoku-very-easy-v0',
         'JobShop-v0']

    To take advante of Jumanji, one usually executes multiple environments at the
    same time.

        >>> from torchrl.envs import JumanjiEnv
        >>> env = JumanjiEnv("Snake-v1", batch_size=[10])
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)

    In the following example, we iteratively test different batch sizes
    and report the execution time for a short rollout:

    Examples:
        >>> from torch.utils.benchmark import Timer
        >>> for batch_size in [4, 16, 128]:
        ...     timer = Timer(
        ...     '''
        ... env.rollout(100)
        ... ''',
        ... setup=f'''
        ... from torchrl.envs import JumanjiEnv
        ... env = JumanjiEnv('Snake-v1', batch_size=[{batch_size}])
        ... env.set_seed(0)
        ... env.rollout(2)
        ... ''')
        ...     print(batch_size, timer.timeit(number=10))
        4 <torch.utils.benchmark.utils.common.Measurement object at 0x1fca91910>
        env.rollout(100)
        setup: [...]
          Median: 122.40 ms
          2 measurements, 1 runs per measurement, 1 thread
        16 <torch.utils.benchmark.utils.common.Measurement object at 0x1ff9baee0>
        env.rollout(100)
        setup: [...]
          Median: 134.39 ms
          2 measurements, 1 runs per measurement, 1 thread
        128 <torch.utils.benchmark.utils.common.Measurement object at 0x1ff9ba7c0>
        env.rollout(100)
        setup: [...]
          Median: 172.31 ms
          2 measurements, 1 runs per measurement, 1 thread
    """

    def __init__(self, env_name, **kwargs):
        kwargs["env_name"] = env_name
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "jumanji.env.Environment":  # noqa: F821
        if not _has_jumanji:
            raise ImportError(
                f"jumanji not found, unable to create {env_name}. "
                f"Consider installing jumanji. More info:"
                f" {self.git_url}."
            )
        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        if kwargs:
            raise ValueError(f"Extra kwargs are not supported by {type(self)}.")
        self.wrapper_frame_skip = 1
        env = self.lib.make(env_name, **kwargs)
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"
