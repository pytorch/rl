# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, TensorDict

from torchrl._utils import logger as torchrl_logger
from torchrl.data import ReplayBuffer, ReplayBufferEnsemble, SamplerWithoutReplacement
from torchrl.data.datasets.atari_dqn import AtariDQNExperienceReplay
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.gen_dgrl import GenDGRLExperienceReplay
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.datasets.openml import OpenMLExperienceReplay
from torchrl.data.datasets.openx import OpenXExperienceReplay
from torchrl.data.datasets.roboset import RobosetExperienceReplay
from torchrl.data.datasets.vd4rl import VD4RLExperienceReplay
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    GrayScale,
    RenameTransform,
    Resize,
    ToTensorImage,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymWrapper, set_gym_backend
from torchrl.envs.libs.openml import OpenMLEnv
from torchrl.envs.utils import check_env_specs
from torchrl.testing import retry

_has_ale_py = importlib.util.find_spec("ale_py") is not None
_has_gymnasium_robotics = importlib.util.find_spec("gymnasium_robotics") is not None

_has_d4rl = importlib.util.find_spec("d4rl") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_has_minari = importlib.util.find_spec("minari") is not None
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None

if importlib.util.find_spec("gym"):
    import gym

if _has_gymnasium:
    import gymnasium


@pytest.mark.slow
class TestGenDGRL:
    @staticmethod
    @pytest.fixture
    def _patch_traj_len():
        # avoids processing the entire dataset
        _get_category_len = GenDGRLExperienceReplay._get_category_len

        def new_get_category_len(cls, category_name):
            return 100

        GenDGRLExperienceReplay._get_category_len = classmethod(new_get_category_len)

        yield
        GenDGRLExperienceReplay._get_category_len = _get_category_len

    @pytest.mark.parametrize("dataset_num", [4])
    def test_gen_dgrl_preproc(self, dataset_num, tmpdir, _patch_traj_len):
        dataset_id = GenDGRLExperienceReplay.available_datasets[dataset_num]
        tmpdir = Path(tmpdir)
        dataset = GenDGRLExperienceReplay(
            dataset_id, batch_size=32, root=tmpdir / "1", download="force"
        )
        t = Compose(
            Resize(32, in_keys=["observation", ("next", "observation")]),
            GrayScale(in_keys=["observation", ("next", "observation")]),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            num_frames=100,
            dest=tmpdir / "2",
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["observation"].shape == torch.Size([32, 1, 32, 32])
        assert sample["next", "observation"].shape == torch.Size([32, 1, 32, 32])

    @pytest.mark.parametrize("dataset_num", [0, 4, 8])
    def test_gen_dgrl(self, dataset_num, tmpdir, _patch_traj_len):
        dataset_id = GenDGRLExperienceReplay.available_datasets[dataset_num]
        dataset = GenDGRLExperienceReplay(dataset_id, batch_size=32, root=tmpdir)
        for batch in dataset:  # noqa: B007
            break
        assert batch.get(("next", "observation")).shape[-3] == 3
        for key in (
            ("next", "done"),
            ("next", "truncated"),
            ("next", "terminated"),
            "observation",
            "action",
            ("next", "reward"),
        ):
            assert key in batch.keys(True, True)
        for key in (
            ("next", "done"),
            ("next", "truncated"),
            ("next", "terminated"),
            "terminated",
            "truncated",
            "done",
            ("next", "reward"),
        ):
            val = batch.get(key)
            assert val.shape[:-1] == batch.shape


@pytest.mark.skipif(not _has_d4rl, reason="D4RL not found")
@pytest.mark.slow
class TestD4RL:
    def test_d4rl_preproc(self, tmpdir):
        dataset_id = "walker2d-medium-replay-v2"
        tmpdir = Path(tmpdir)
        dataset = D4RLExperienceReplay(
            dataset_id,
            batch_size=32,
            root=tmpdir / "1",
            download="force",
            direct_download=True,
        )
        t = Compose(
            CatTensors(
                in_keys=["observation", ("info", "qpos"), ("info", "qvel")],
                out_key="data",
            ),
            CatTensors(
                in_keys=[
                    ("next", "observation"),
                    ("next", "info", "qpos"),
                    ("next", "info", "qvel"),
                ],
                out_key=("next", "data"),
            ),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir / "2",
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["data"].shape == torch.Size([32, 35])
        assert sample["next", "data"].shape == torch.Size([32, 35])

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("use_truncated_as_done", [True, False])
    @pytest.mark.parametrize("split_trajs", [True, False])
    def test_terminate_on_end(self, task, use_truncated_as_done, split_trajs, tmpdir):
        root1 = tmpdir / "1"
        root2 = tmpdir / "2"
        root3 = tmpdir / "3"

        with (
            pytest.warns(UserWarning, match="Using use_truncated_as_done=True")
            if use_truncated_as_done
            else nullcontext()
        ):
            data_true = D4RLExperienceReplay(
                task,
                split_trajs=split_trajs,
                from_env=False,
                terminate_on_end=True,
                batch_size=2,
                use_truncated_as_done=use_truncated_as_done,
                download="force",
                root=root1,
            )
        _ = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=False,
            terminate_on_end=False,
            batch_size=2,
            use_truncated_as_done=use_truncated_as_done,
            download="force",
            root=root2,
        )
        data_from_env = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=True,
            batch_size=2,
            use_truncated_as_done=use_truncated_as_done,
            download="force",
            root=root3,
        )
        if not use_truncated_as_done:
            keys = set(data_from_env[:].keys(True, True))
            keys = keys.intersection(data_true[:].keys(True, True))
            assert data_true[:].shape == data_from_env[:].shape
            # for some reason, qlearning_dataset overwrites the next obs that is contained in the buffer,
            # resulting in tiny changes in the value contained for that key. Over 99.99% of the values
            # match, but the test still fails because of this.
            # We exclude that entry from the comparison.
            keys.discard(("next", "observation"))
            assert_allclose_td(
                data_true[:].select(*keys),
                data_from_env[:].select(*keys),
            )
        else:
            leaf_names = data_from_env[:].keys(True)
            leaf_names = [
                name[-1] if isinstance(name, tuple) else name for name in leaf_names
            ]
            assert "truncated" in leaf_names
            leaf_names = data_true[:].keys(True)
            leaf_names = [
                name[-1] if isinstance(name, tuple) else name for name in leaf_names
            ]
            assert "truncated" not in leaf_names

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    def test_direct_download(self, task, tmpdir):
        root1 = tmpdir / "1"
        root2 = tmpdir / "2"
        data_direct = D4RLExperienceReplay(
            task,
            split_trajs=False,
            from_env=False,
            batch_size=2,
            use_truncated_as_done=True,
            direct_download=True,
            download="force",
            root=root1,
        )
        data_d4rl = D4RLExperienceReplay(
            task,
            split_trajs=False,
            from_env=True,
            batch_size=2,
            use_truncated_as_done=True,
            direct_download=False,
            terminate_on_end=True,  # keep the last time step
            download="force",
            root=root2,
        )
        keys = set(data_direct[:].keys(True, True))
        keys = keys.intersection(data_d4rl[:].keys(True, True))
        assert len(keys)
        assert_allclose_td(
            data_direct[:].select(*keys).apply(lambda t: t.float()),
            data_d4rl[:].select(*keys).apply(lambda t: t.float()),
        )

    @pytest.mark.parametrize(
        "task",
        [
            # "antmaze-medium-play-v0",
            # "hammer-cloned-v1",
            # "maze2d-open-v0",
            # "maze2d-open-dense-v0",
            # "relocate-human-v1",
            "walker2d-medium-replay-v2",
            # "ant-medium-v2",
            # # "flow-merge-random-v0",
            # "kitchen-partial-v0",
            # # "carla-town-v0",
        ],
    )
    def test_d4rl_dummy(self, task):
        t0 = time.time()
        _ = D4RLExperienceReplay(task, split_trajs=True, from_env=True, batch_size=2)
        torchrl_logger.info(f"terminated test after {time.time() - t0}s")

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("split_trajs", [True, False])
    @pytest.mark.parametrize("from_env", [True, False])
    def test_dataset_build(self, task, split_trajs, from_env):
        import d4rl  # noqa: F401

        t0 = time.time()
        data = D4RLExperienceReplay(
            task, split_trajs=split_trajs, from_env=from_env, batch_size=2
        )
        sample = data.sample()
        # D4RL environments are registered with gym, not gymnasium
        with set_gym_backend("gym"):
            env = GymWrapper(gym.make(task))
        rollout = env.rollout(2)
        for key in rollout.keys(True, True):
            if "truncated" in key:
                # truncated is missing from static datasets
                continue
            sim = rollout.get(key)
            offline = sample.get(key)
            # assert sim.dtype == offline.dtype, key
            assert sim.shape[-1] == offline.shape[-1], key
        torchrl_logger.info(f"terminated test after {time.time() - t0}s")

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("split_trajs", [True, False])
    def test_d4rl_iteration(self, task, split_trajs):
        t0 = time.time()
        batch_size = 3
        data = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=False,
            terminate_on_end=True,
            batch_size=batch_size,
            sampler=SamplerWithoutReplacement(drop_last=True),
        )
        i = 0
        for sample in data:  # noqa: B007
            i += 1
        assert len(data) // i == batch_size
        torchrl_logger.info(f"terminated test after {time.time() - t0}s")


_MINARI_DATASETS = []

MUJOCO_ENVIRONMENTS = [
    "Hopper-v5",
    "Pusher-v4",
    "Humanoid-v5",
    "InvertedDoublePendulum-v5",
    "HalfCheetah-v5",
    "Swimmer-v5",
    "Walker2d-v5",
    "ALE/Ant-v5",
    "Reacher-v5",
]

D4RL_ENVIRONMENTS = [
    "AntMaze_UMaze-v5",
    "AdroitHandPen-v1",
    "AntMaze_Medium-v4",
    "AntMaze_Large_Diverse_GR-v4",
    "AntMaze_Large-v4",
    "AntMaze_Medium_Diverse_GR-v4",
    "PointMaze_OpenDense-v3",
    "PointMaze_UMaze-v3",
    "PointMaze_LargeDense-v3",
    "PointMaze_Medium-v3",
    "PointMaze_UMazeDense-v3",
    "PointMaze_MediumDense-v3",
    "PointMaze_Large-v3",
    "PointMaze_Open-v3",
    "FrankaKitchen-v1",
    "AdroitHandDoor-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
]

MUJOCO_ENVIRONMENTS = [
    "Hopper-v5",
    "Pusher-v5",
    "Humanoid-v5",
    "InvertedDoublePendulum-v5",
    "HalfCheetah-v5",
    "Swimmer-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Reacher-v5",
]

D4RL_ENVIRONMENTS = [
    "AntMaze_UMaze-v5",
    "AdroitHandPen-v1",
    "AntMaze_Medium-v4",
    "AntMaze_Large_Diverse_GR-v4",
    "AntMaze_Large-v4",
    "AntMaze_Medium_Diverse_GR-v4",
    "PointMaze_OpenDense-v3",
    "PointMaze_UMaze-v3",
    "PointMaze_LargeDense-v3",
    "PointMaze_Medium-v3",
    "PointMaze_UMazeDense-v3",
    "PointMaze_MediumDense-v3",
    "PointMaze_Large-v3",
    "PointMaze_Open-v3",
    "FrankaKitchen-v1",
    "AdroitHandDoor-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
]


def _minari_init() -> tuple[bool, Exception | None]:
    """Initialize Minari datasets list. Returns True if already initialized."""
    global _MINARI_DATASETS  # noqa: F824
    if _MINARI_DATASETS and not all(
        isinstance(x, str) and x.isdigit() for x in _MINARI_DATASETS
    ):
        return True, None  # Already initialized with real dataset names

    if not _has_minari or not _has_gymnasium:
        return False, ImportError("Minari or Gymnasium not found")

    try:
        import minari

        torch.manual_seed(0)

        total_keys = sorted(
            minari.list_remote_datasets(
                latest_version=True, compatible_minari_version=True
            )
        )
        indices = torch.randperm(len(total_keys))[:20]
        keys = [total_keys[idx] for idx in indices]

        assert len(keys) > 5, keys
        _MINARI_DATASETS[:] = keys  # Replace the placeholder values
        return True, None
    except Exception as err:
        return False, err


def get_random_minigrid_datasets():
    """
    Fetch 5 random Minigrid datasets from the Minari server.
    """
    import minari

    all_minigrid = [
        dataset
        for dataset in minari.list_remote_datasets(
            latest_version=True, compatible_minari_version=True
        ).keys()
        if dataset.startswith("minigrid/")
    ]

    # 3 random datasets
    indices = torch.randperm(len(all_minigrid))[:3]
    return [all_minigrid[idx] for idx in indices]


def get_random_atari_envs():
    """
    Fetch 3 random Atari environments using ale_py and torch.
    """
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    env_specs = gym.envs.registry.values()
    all_env_ids = [env_spec.id for env_spec in env_specs]
    atari_env_ids = [env_id for env_id in all_env_ids if env_id.startswith("ALE")]
    if len(atari_env_ids) < 3:
        raise RuntimeError("Not enough Atari environments found.")
    indices = torch.randperm(len(atari_env_ids))[:3]
    return [atari_env_ids[idx] for idx in indices]


def custom_minari_init(custom_envs, num_episodes=5):
    """
    Initialize custom Minari datasets for the given environments.
    """
    import gymnasium
    import gymnasium_robotics
    from minari import DataCollector

    gymnasium.register_envs(gymnasium_robotics)

    custom_dataset_ids = []
    for env_id in custom_envs:
        dataset_id = f"{env_id.lower()}/test-custom-local-v1"
        env = gymnasium.make(env_id)
        collector = DataCollector(env)

        for ep in range(num_episodes):
            collector.reset(seed=123 + ep)

            while True:
                action = collector.action_space.sample()
                _, _, terminated, truncated, _ = collector.step(action)
                if terminated or truncated:
                    break

        collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="RandomPolicy",
            code_permalink="https://github.com/Farama-Foundation/Minari",
            author="Farama",
            author_email="contact@farama.org",
            eval_env=env_id,
        )
        custom_dataset_ids.append(dataset_id)

    return custom_dataset_ids


@pytest.mark.skipif(not _has_minari or not _has_gymnasium, reason="Minari not found")
@pytest.mark.slow
class TestMinari:
    @pytest.mark.parametrize("split", [False, True])
    @pytest.mark.parametrize(
        "dataset_idx",
        # Only use a static upper bound; do not call any function that imports minari globally.
        range(4),
    )
    def test_load(self, dataset_idx, split):
        """
        Test loading from custom datasets for Mujoco and D4RL,
        Minari remote datasets for Minigrid, and random Atari environments.
        """
        import minari

        num_custom_to_select = 4
        custom_envs = MUJOCO_ENVIRONMENTS + D4RL_ENVIRONMENTS

        # Randomly select a subset of custom environments
        indices = torch.randperm(len(custom_envs))[:num_custom_to_select]
        custom_envs_subset = [custom_envs[i] for i in indices]

        num_custom = len(custom_envs_subset)
        try:
            minigrid_datasets = get_random_minigrid_datasets()
        except Exception:
            minigrid_datasets = []
        num_minigrid = len(minigrid_datasets)
        try:
            atari_envs = get_random_atari_envs()
        except Exception:
            atari_envs = []
        num_atari = len(atari_envs)
        total_datasets = num_custom + num_minigrid + num_atari

        if dataset_idx >= total_datasets:
            pytest.skip("Index out of range for available datasets")

        if dataset_idx < num_custom:
            # Custom dataset for Mujoco/D4RL
            custom_dataset_ids = custom_minari_init(
                [custom_envs_subset[dataset_idx]], num_episodes=5
            )
            dataset_id = custom_dataset_ids[0]
            data = MinariExperienceReplay(
                dataset_id=dataset_id,
                split_trajs=split,
                batch_size=32,
                load_from_local_minari=True,
            )
            cleanup_needed = True

        elif dataset_idx < num_custom + num_minigrid:
            # Minigrid datasets from Minari server
            minigrid_idx = dataset_idx - num_custom
            dataset_id = minigrid_datasets[minigrid_idx]
            data = MinariExperienceReplay(
                dataset_id=dataset_id,
                batch_size=32,
                split_trajs=split,
                download="force",
            )
            cleanup_needed = False

        else:
            # Atari environment datasets
            atari_idx = dataset_idx - num_custom - num_minigrid
            env_id = atari_envs[atari_idx]
            custom_dataset_ids = custom_minari_init([env_id], num_episodes=5)
            dataset_id = custom_dataset_ids[0]
            data = MinariExperienceReplay(
                dataset_id=dataset_id,
                split_trajs=split,
                batch_size=32,
                load_from_local_minari=True,
            )
            cleanup_needed = True

        t0 = time.time()
        for i, sample in enumerate(data):
            t1 = time.time()
            torchrl_logger.info(f"sampling time {1000 * (t1 - t0): 4.4f}ms")
            assert data.metadata["action_space"].is_in(sample["action"])
            assert data.metadata["observation_space"].is_in(sample["observation"])
            t0 = time.time()
            if i == 10:
                break

        # Clean up custom datasets after running local dataset tests
        if cleanup_needed:
            minari.delete_dataset(dataset_id=dataset_id)

    @retry(Exception, tries=3, delay=1)
    def test_minari_preproc(self, tmpdir):
        dataset = MinariExperienceReplay(
            "D4RL/pointmaze/large-v2",
            batch_size=32,
            split_trajs=False,
            download="force",
        )

        t = Compose(
            CatTensors(
                in_keys=[
                    ("observation", "observation"),
                    ("info", "qpos"),
                    ("info", "qvel"),
                ],
                out_key="data",
            ),
            CatTensors(
                in_keys=[
                    ("next", "observation", "observation"),
                    ("next", "info", "qpos"),
                    ("next", "info", "qvel"),
                ],
                out_key=("next", "data"),
            ),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            num_frames=100,
            dest=tmpdir,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        sample = dataset.sample()
        assert len(dataset) == 100
        assert sample["data"].shape == torch.Size([32, 8])
        assert sample["next", "data"].shape == torch.Size([32, 8])

    @pytest.mark.skipif(
        not _has_minari or not _has_gymnasium, reason="Minari or Gym not available"
    )
    def test_local_minari_dataset_loading(self, tmpdir):
        MINARI_DATASETS_PATH = os.environ.get("MINARI_DATASETS_PATH")
        os.environ["MINARI_DATASETS_PATH"] = str(tmpdir)
        try:
            import minari
            from minari import DataCollector

            success, err = _minari_init()
            if not success:
                pytest.skip(f"Failed to initialize Minari datasets: {err}")

            dataset_id = "cartpole/test-local-v1"

            # Create dataset using Gym + DataCollector
            env = gymnasium.make("CartPole-v1")
            env = DataCollector(env, record_infos=True)
            for _ in range(50):
                env.reset(seed=123)
                while True:
                    action = env.action_space.sample()
                    obs, rew, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

            env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="RandomPolicy",
                code_permalink="https://github.com/Farama-Foundation/Minari",
                author="Farama",
                author_email="contact@farama.org",
                eval_env="CartPole-v1",
            )

            # Load from local cache
            data = MinariExperienceReplay(
                dataset_id=dataset_id,
                split_trajs=False,
                batch_size=32,
                download=False,
                sampler=SamplerWithoutReplacement(drop_last=True),
                prefetch=2,
                load_from_local_minari=True,
            )

            t0 = time.time()
            for i, sample in enumerate(data):
                t1 = time.time()
                torchrl_logger.info(
                    f"[Local Minari] Sampling time {1000 * (t1 - t0):4.4f} ms"
                )
                assert data.metadata["action_space"].is_in(
                    sample["action"]
                ), "Invalid action sample"
                assert data.metadata["observation_space"].is_in(
                    sample["observation"]
                ), "Invalid observation sample"
                t0 = time.time()
                if i == 10:
                    break

            minari.delete_dataset(dataset_id="cartpole/test-local-v1")
        finally:
            if MINARI_DATASETS_PATH:
                os.environ["MINARI_DATASETS_PATH"] = MINARI_DATASETS_PATH

    def test_correct_categorical_missions(self):
        try:
            exp_replay = MinariExperienceReplay(
                dataset_id="minigrid/BabyAI-Pickup/optimal-v0",
                batch_size=1,
                root=None,
            )
        except Exception as e:
            err_str = str(e).lower()
            if any(
                x in err_str
                for x in (
                    "429",
                    "too many requests",
                    "not found locally",
                    "download failed",
                )
            ):
                warnings.warn(f"Test inconclusive due to download failure: {e}")
                return
            raise
        assert isinstance(exp_replay[0][("observation", "mission")], (bytes, str))


@pytest.mark.slow
class TestRoboset:
    def test_load(self):
        selected_dataset = RobosetExperienceReplay.available_datasets[0]
        data = RobosetExperienceReplay(
            selected_dataset,
            batch_size=32,
        )
        t0 = time.time()
        for i, _ in enumerate(data):
            t1 = time.time()
            torchrl_logger.info(f"sampling time {1000 * (t1 - t0): 4.4f}ms")
            t0 = time.time()
            if i == 10:
                break

    def test_roboset_preproc(self, tmpdir):
        dataset = RobosetExperienceReplay(
            "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4", batch_size=32, download="force"
        )

        def func(data):
            return data.set("obs_norm", data.get("observation").norm(dim=-1))

        new_storage = dataset.preprocess(
            func,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir,
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert "obs_norm" in sample.keys()


@pytest.mark.slow
class TestVD4RL:
    @pytest.mark.parametrize("image_size", [None, (37, 33)])
    def test_load(self, image_size):
        torch.manual_seed(0)
        datasets = VD4RLExperienceReplay.available_datasets
        for idx in torch.randperm(len(datasets)).tolist()[:4]:
            selected_dataset = datasets[idx]
            data = VD4RLExperienceReplay(
                selected_dataset,
                batch_size=32,
                image_size=image_size,
            )
            t0 = time.time()
            for i, batch in enumerate(data):
                if image_size:
                    assert batch.get("pixels").shape == (32, 3, *image_size)
                    assert batch.get(("next", "pixels")).shape == (32, 3, *image_size)
                else:
                    assert batch.get("pixels").shape[:2] == (32, 3)
                    assert batch.get(("next", "pixels")).shape[:2] == (32, 3)

                assert batch.get("pixels").dtype is torch.float32
                assert batch.get(("next", "pixels")).dtype is torch.float32
                assert (batch.get("pixels") != 0).any()
                assert (batch.get(("next", "pixels")) != 0).any()
                t1 = time.time()
                torchrl_logger.info(f"sampling time {1000 * (t1 - t0): 4.4f}ms")
                t0 = time.time()
                if i == 10:
                    break

    def test_vd4rl_preproc(self, tmpdir):
        torch.manual_seed(0)
        datasets = VD4RLExperienceReplay.available_datasets
        dataset_id = list(datasets)[4]
        dataset = VD4RLExperienceReplay(dataset_id, batch_size=32, download="force")
        func = Compose(
            ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
            GrayScale(in_keys=["pixels", ("next", "pixels")]),
        )
        new_storage = dataset.preprocess(
            func,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir,
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["next", "pixels"].shape == torch.Size([32, 1, 64, 64])


@pytest.mark.slow
class TestAtariDQN:
    @pytest.fixture(scope="class")
    def limit_max_runs(self):
        prev_val = AtariDQNExperienceReplay._max_runs
        AtariDQNExperienceReplay._max_runs = 3
        yield
        AtariDQNExperienceReplay._max_runs = prev_val

    @pytest.mark.parametrize("dataset_id", ["Asterix/1", "Pong/4"])
    @pytest.mark.parametrize(
        "num_slices,slice_len", [[None, None], [None, 8], [2, None]]
    )
    def test_single_dataset(self, dataset_id, slice_len, num_slices, limit_max_runs):
        dataset = AtariDQNExperienceReplay(
            dataset_id, slice_len=slice_len, num_slices=num_slices
        )
        sample = dataset.sample(64)
        for key in (
            ("next", "observation"),
            ("next", "truncated"),
            ("next", "terminated"),
            ("next", "done"),
            ("next", "reward"),
            "observation",
            "action",
            "done",
            "truncated",
            "terminated",
        ):
            assert key in sample.keys(True)
        assert sample.shape == (64,)
        assert sample.get_non_tensor("metadata")["dataset_id"] == dataset_id

    @pytest.mark.parametrize(
        "num_slices,slice_len", [[None, None], [None, 8], [2, None]]
    )
    def test_double_dataset(self, slice_len, num_slices, limit_max_runs):
        dataset_pong = AtariDQNExperienceReplay(
            "Pong/4", slice_len=slice_len, num_slices=num_slices
        )
        dataset_asterix = AtariDQNExperienceReplay(
            "Asterix/1", slice_len=slice_len, num_slices=num_slices
        )
        dataset = ReplayBufferEnsemble(
            dataset_pong, dataset_asterix, sample_from_all=True, batch_size=128
        )
        sample = dataset.sample()
        assert sample.shape == (2, 64)
        assert sample[0].get_non_tensor("metadata")["dataset_id"] == "Pong/4"
        assert sample[1].get_non_tensor("metadata")["dataset_id"] == "Asterix/1"

    @pytest.mark.parametrize("dataset_id", ["Pong/4"])
    def test_atari_preproc(self, dataset_id, tmpdir):
        dataset = AtariDQNExperienceReplay(
            dataset_id,
            slice_len=None,
            num_slices=8,
            batch_size=64,
            # num_procs=max(0, os.cpu_count() - 4),
            num_procs=0,
        )

        t = Compose(
            UnsqueezeTransform(
                dim=-3, in_keys=["observation", ("next", "observation")]
            ),
            Resize(32, in_keys=["observation", ("next", "observation")]),
            RenameTransform(in_keys=["action"], out_keys=["other_action"]),
        )

        def preproc(data):
            return t(data)

        new_storage = dataset.preprocess(
            preproc,
            num_workers=max(1, os.cpu_count() - 4),
            num_chunks=1000,
            # mp_start_method="fork",
            pbar=True,
            dest=tmpdir,
            num_frames=100,
        )

        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100


@pytest.mark.slow
class TestOpenX:
    @pytest.mark.parametrize(
        "download,padding",
        [[True, None], [False, None], [False, 0], [False, True], [False, False]],
    )
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("replacement", [True, False])
    @pytest.mark.parametrize(
        "batch_size,num_slices,slice_len",
        [
            [3000, 2, None],
            [32, 32, None],
            [32, None, 1],
            [3000, None, 1500],
            [None, None, 32],
            [None, None, 1500],
        ],
    )
    def test_openx(
        self, download, shuffle, replacement, padding, batch_size, num_slices, slice_len
    ):
        torch.manual_seed(0)
        np.random.seed(0)

        streaming = not download
        cm = (
            pytest.raises(RuntimeError, match="shuffle=False")
            if not streaming and not shuffle and replacement
            else pytest.raises(
                RuntimeError,
                match="replacement=True is not available with streamed datasets",
            )
            if streaming and replacement
            else nullcontext()
        )
        dataset = None
        with cm:
            dataset = OpenXExperienceReplay(
                "cmu_stretch",
                download=download,
                streaming=streaming,
                batch_size=batch_size,
                shuffle=shuffle,
                num_slices=num_slices,
                slice_len=slice_len,
                pad=padding,
                replacement=replacement,
            )
        if dataset is None:
            return
        # iterating
        if padding is None and (
            (batch_size is not None and batch_size > 1000)
            or (slice_len is not None and slice_len > 1000)
        ):
            raises_cm = pytest.raises(
                RuntimeError,
                match="The trajectory length (.*) is shorter than the slice length|"
                #       "Some stored trajectories have a length shorter than the slice that was asked for|"
                "Did not find a single trajectory with sufficient length",
            )
            with raises_cm:
                for data in dataset:  # noqa: B007
                    break
            if batch_size is None and slice_len is not None:
                with raises_cm:
                    dataset.sample(2 * slice_len)
                return

        else:
            for data in dataset:  # noqa: B007
                break
            # check data shape
            if batch_size is not None:
                assert data.shape[0] == batch_size
            elif slice_len is not None:
                assert data.shape[0] == slice_len
            if batch_size is not None:
                if num_slices is not None:
                    assert data.get(("next", "done")).sum(-2) == num_slices
                elif streaming:
                    assert (
                        data.get(("next", "done")).sum(-2)
                        == data.get("episode").unique().numel()
                    )

        # sampling
        if batch_size is None:
            if slice_len is not None:
                batch_size = 2 * slice_len
            elif num_slices is not None:
                batch_size = num_slices * 32
            sample = dataset.sample(batch_size)
        else:
            if padding is None and (batch_size > 1000):
                with pytest.raises(
                    RuntimeError,
                    match="Did not find a single trajectory with sufficient length"
                    if not streaming
                    else "The trajectory length (.*) is shorter than the slice length",
                ):
                    sample = dataset.sample()
                return
            else:
                sample = dataset.sample()
                assert sample.shape == (batch_size,)
        if slice_len is not None:
            assert sample.get(("next", "done")).sum() == int(
                batch_size // slice_len
            ), sample.get(("next", "done"))
        elif num_slices is not None:
            assert sample.get(("next", "done")).sum() == num_slices

    def test_openx_preproc(self, tmpdir):
        dataset = OpenXExperienceReplay(
            "cmu_stretch",
            download=True,
            streaming=False,
            batch_size=64,
            shuffle=True,
            num_slices=8,
            slice_len=None,
        )
        t = Compose(
            Resize(
                64,
                64,
                in_keys=[("observation", "image"), ("next", "observation", "image")],
            ),
            RenameTransform(
                in_keys=[
                    ("observation", "image"),
                    ("next", "observation", "image"),
                    ("observation", "state"),
                    ("next", "observation", "state"),
                ],
                out_keys=["pixels", ("next", "pixels"), "state", ("next", "state")],
            ),
        )

        def fn(data: TensorDict):
            data.unlock_()
            data = data.select(
                "action",
                "done",
                "episode",
                ("next", "done"),
                ("next", "observation"),
                ("next", "reward"),
                ("next", "terminated"),
                ("next", "truncated"),
                "observation",
                "terminated",
                "truncated",
            )
            data = t(data)
            data = data.select(*data.keys(True, True))
            return data

        new_storage = dataset.preprocess(
            CloudpickleWrapper(fn),
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=500,
            # mp_start_method="fork",
            dest=tmpdir,
        )
        dataset = ReplayBuffer(storage=new_storage)
        sample = dataset.sample(32)
        assert "observation" not in sample.keys()
        assert "pixels" in sample.keys()
        assert ("next", "pixels") in sample.keys(True)
        assert "state" in sample.keys()
        assert ("next", "state") in sample.keys(True)
        assert sample["pixels"].shape == torch.Size([32, 3, 64, 64])


@pytest.mark.skipif(not _has_sklearn, reason="Scikit-learn not found")
@pytest.mark.parametrize(
    "dataset",
    [
        # "adult_num", # 1226: Expensive to test
        # "adult_onehot", # 1226: Expensive to test
        "mushroom_num",
        "mushroom_onehot",
        # "covertype",  # 1226: Expensive to test
        "shuttle",
        "magic",
    ],
)
@pytest.mark.slow
class TestOpenML:
    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 3)])
    def test_env(self, dataset, batch_size):
        env = OpenMLEnv(dataset, batch_size=batch_size)
        td = env.reset()
        assert td.shape == torch.Size(batch_size)
        td = env.rand_step(td)
        assert td.shape == torch.Size(batch_size)
        assert "index" not in td.keys()
        check_env_specs(env)

    def test_data(self, dataset):
        data = OpenMLExperienceReplay(
            dataset,
            batch_size=2048,
            transform=Compose(
                RenameTransform(["X"], ["observation"]),
                DoubleToFloat(["observation"]),
            ),
        )
        # check that dataset eventually runs out
        for i, _ in enumerate(data):  # noqa: B007
            continue
        assert len(data) // 2048 in (i, i - 1)
