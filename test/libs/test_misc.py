# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import urllib
import urllib.error
from unittest import mock

import pytest
import torch
import torch.nn as nn
from tensordict import is_tensor_collection, LazyStackedTensorDict, TensorDict
from tensordict.nn import TensorDictModule

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.distributed import RayEvalWorker
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.libs.openspiel import _has_pyspiel, OpenSpielEnv, OpenSpielWrapper
from torchrl.envs.libs.procgen import ProcgenEnv, ProcgenWrapper
from torchrl.envs.libs.robohive import _has_robohive, RoboHiveEnv
from torchrl.envs.libs.unity_mlagents import (
    _has_unity_mlagents,
    UnityMLAgentsEnv,
    UnityMLAgentsWrapper,
)
from torchrl.envs.utils import check_env_specs, MarlGroupMapType
from torchrl.testing import retry

_has_mlagents_envs = importlib.util.find_spec("mlagents_envs") is not None

_has_ray = importlib.util.find_spec("ray") is not None
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
_has_procgen = importlib.util.find_spec("procgen") is not None


@pytest.mark.skipif(not _has_robohive, reason="RoboHive not found")
class TestRoboHive:
    # unfortunately we must import robohive to get the available envs
    # and this import will occur whenever pytest is run on this file.
    # The other option would be not to use parametrize but that also
    # means less informative error trace stacks.
    # In the CI, robohive should not coexist with other libs so that's fine.
    # Robohive logging behavior can be controlled via ROBOHIVE_VERBOSITY=ALL/INFO/(WARN)/ERROR/ONCE/ALWAYS/SILENT
    @pytest.mark.parametrize("from_pixels", [False, True])
    @pytest.mark.parametrize("from_depths", [False, True])
    @pytest.mark.parametrize("envname", RoboHiveEnv.available_envs)
    def test_robohive(self, envname, from_pixels, from_depths):
        with set_gym_backend("gymnasium"):
            torchrl_logger.info(f"{envname}-{from_pixels}-{from_depths}")
            if any(
                substr in envname for substr in ("_vr3m", "_vrrl", "_vflat", "_vvc1s")
            ):
                torchrl_logger.info("not testing envs with prebuilt rendering")
                return
            if "Adroit" in envname:
                torchrl_logger.info("tcdm are broken")
                return
            if (
                from_pixels
                and len(RoboHiveEnv.get_available_cams(env_name=envname)) == 0
            ):
                torchrl_logger.info("no camera")
                return
            try:
                env = RoboHiveEnv(
                    envname, from_pixels=from_pixels, from_depths=from_depths
                )
            except AttributeError as err:
                if "'MjData' object has no attribute 'get_body_xipos'" in str(err):
                    torchrl_logger.info("tcdm are broken")
                    return
                else:
                    raise err
            # Make sure that the stack is dense
            for val in env.rollout(4).values(True):
                if is_tensor_collection(val):
                    assert not isinstance(val, LazyStackedTensorDict)
                    assert not val.is_empty()
            check_env_specs(env)


# List of OpenSpiel games to test
# TODO: Some of the games in `OpenSpielWrapper.available_envs` raise errors for
# a few different reasons, mostly because we do not support chance nodes yet. So
# we cannot run tests on all of them yet.
_openspiel_games = [
    # ----------------
    # Sequential games
    # 1-player
    "morpion_solitaire",
    # 2-player
    "amazons",
    "battleship",
    "breakthrough",
    "checkers",
    "chess",
    "cliff_walking",
    "clobber",
    "connect_four",
    "cursor_go",
    "dark_chess",
    "dark_hex",
    "dark_hex_ir",
    "dots_and_boxes",
    "go",
    "havannah",
    "hex",
    "kriegspiel",
    "mancala",
    "nim",
    "nine_mens_morris",
    "othello",
    "oware",
    "pentago",
    "phantom_go",
    "phantom_ttt",
    "phantom_ttt_ir",
    "sheriff",
    "tic_tac_toe",
    "twixt",
    "ultimate_tic_tac_toe",
    "y",
    # --------------
    # Parallel games
    # 2-player
    "blotto",
    "matrix_bos",
    "matrix_brps",
    "matrix_cd",
    "matrix_coordination",
    "matrix_mp",
    "matrix_pd",
    "matrix_rps",
    "matrix_rpsw",
    "matrix_sh",
    "matrix_shapleys_game",
    "oshi_zumo",
    # 3-player
    "matching_pennies_3p",
]


@pytest.mark.skipif(not _has_pyspiel, reason="open_spiel not found")
class TestOpenSpiel:
    @pytest.mark.parametrize("game_string", _openspiel_games)
    @pytest.mark.parametrize("return_state", [False, True])
    @pytest.mark.parametrize("categorical_actions", [False, True])
    def test_all_envs(self, game_string, return_state, categorical_actions):
        env = OpenSpielEnv(
            game_string,
            categorical_actions=categorical_actions,
            return_state=return_state,
        )
        check_env_specs(env)

    @pytest.mark.parametrize("game_string", _openspiel_games)
    @pytest.mark.parametrize("return_state", [False, True])
    @pytest.mark.parametrize("categorical_actions", [False, True])
    def test_wrapper(self, game_string, return_state, categorical_actions):
        import pyspiel

        base_env = pyspiel.load_game(game_string).new_initial_state()
        env_torchrl = OpenSpielWrapper(
            base_env, categorical_actions=categorical_actions, return_state=return_state
        )
        env_torchrl.rollout(max_steps=5)

    @pytest.mark.parametrize("game_string", _openspiel_games)
    @pytest.mark.parametrize("return_state", [False, True])
    @pytest.mark.parametrize("categorical_actions", [False, True])
    def test_reset_state(self, game_string, return_state, categorical_actions):
        env = OpenSpielEnv(
            game_string,
            categorical_actions=categorical_actions,
            return_state=return_state,
        )
        td = env.reset()
        td_init = td.clone()

        # Perform an action
        td = env.step(env.full_action_spec.rand())

        # Save the current td for reset
        td_reset = td["next"].clone()

        # Perform a second action
        td = env.step(env.full_action_spec.rand())

        # Resetting to a specific state can only happen if `return_state` is
        # enabled. Otherwise, it is reset to the initial state.
        if return_state:
            # Check that the state was reset to the specified state
            td = env.reset(td_reset)
            assert (td == td_reset).all()
        else:
            # Check that the state was reset to the initial state
            td = env.reset()
            assert (td == td_init).all()

    def test_chance_not_implemented(self):
        with pytest.raises(
            NotImplementedError,
            match="not yet supported",
        ):
            OpenSpielEnv("bridge")


# NOTE: Each of the registered envs are around 180 MB, so only test a few.
_mlagents_registered_envs = [
    "3DBall",
    "StrikersVsGoalie",
]


@pytest.mark.skipif(not _has_unity_mlagents, reason="mlagents_envs not found")
class TestUnityMLAgents:
    @mock.patch("mlagents_envs.env_utils.launch_executable")
    @mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ONE_GROUP_PER_AGENT, MarlGroupMapType.ALL_IN_ONE_GROUP],
    )
    def test_env(self, mock_communicator, mock_launcher, group_map):
        from mlagents_envs.mock_communicator import MockCommunicator

        mock_communicator.return_value = MockCommunicator(
            discrete_action=False, visual_inputs=0
        )
        env = UnityMLAgentsEnv(" ", group_map=group_map)
        try:
            check_env_specs(env)
        finally:
            env.close()

    @mock.patch("mlagents_envs.env_utils.launch_executable")
    @mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ONE_GROUP_PER_AGENT, MarlGroupMapType.ALL_IN_ONE_GROUP],
    )
    def test_wrapper(self, mock_communicator, mock_launcher, group_map):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.mock_communicator import MockCommunicator

        mock_communicator.return_value = MockCommunicator(
            discrete_action=False, visual_inputs=0
        )
        env = UnityMLAgentsWrapper(UnityEnvironment(" "), group_map=group_map)
        try:
            check_env_specs(env)
        finally:
            env.close()

    @mock.patch("mlagents_envs.env_utils.launch_executable")
    @mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ONE_GROUP_PER_AGENT, MarlGroupMapType.ALL_IN_ONE_GROUP],
    )
    def test_rollout(self, mock_communicator, mock_launcher, group_map):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.mock_communicator import MockCommunicator

        mock_communicator.return_value = MockCommunicator(
            discrete_action=False, visual_inputs=0
        )
        env = UnityMLAgentsWrapper(UnityEnvironment(" "), group_map=group_map)
        try:
            env.rollout(
                max_steps=500, break_when_any_done=False, break_when_all_done=False
            )
        finally:
            env.close()

    @pytest.mark.unity_editor
    def test_with_editor(self):
        print("Please press play in the Unity editor")  # noqa: T201
        env = UnityMLAgentsEnv(timeout_wait=30)
        try:
            env.reset()
            check_env_specs(env)

            # Perform a rollout
            td = env.reset()
            env.rollout(
                max_steps=100, break_when_any_done=False, break_when_all_done=False
            )

            # Step manually
            tensordicts = []
            td = env.reset()
            tensordicts.append(td)
            traj_len = 200
            for _ in range(traj_len - 1):
                td = env.step(td.update(env.full_action_spec.rand()))
                tensordicts.append(td)

            traj = torch.stack(tensordicts)
            assert traj.batch_size == torch.Size([traj_len])
        finally:
            env.close()

    @retry(
        (
            urllib.error.HTTPError,
            urllib.error.URLError,
            urllib.error.ContentTooShortError,
        ),
        5,
    )
    @pytest.mark.parametrize("registered_name", _mlagents_registered_envs)
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ONE_GROUP_PER_AGENT, MarlGroupMapType.ALL_IN_ONE_GROUP],
    )
    def test_registered_envs(self, registered_name, group_map):
        env = UnityMLAgentsEnv(
            registered_name=registered_name,
            no_graphics=True,
            group_map=group_map,
        )
        try:
            check_env_specs(env)

            # Perform a rollout
            td = env.reset()
            env.rollout(
                max_steps=20, break_when_any_done=False, break_when_all_done=False
            )

            # Step manually
            tensordicts = []
            td = env.reset()
            tensordicts.append(td)
            traj_len = 20
            for _ in range(traj_len - 1):
                td = env.step(td.update(env.full_action_spec.rand()))
                tensordicts.append(td)

            traj = torch.stack(tensordicts)
            assert traj.batch_size == torch.Size([traj_len])
        finally:
            env.close()


@pytest.mark.skipif(
    not _has_ray or not _has_gymnasium, reason="Ray or Gymnasium not found"
)
class TestRayEvalWorker:
    """Tests for the RayEvalWorker async evaluation helper."""

    @pytest.fixture(autouse=True)
    def _setup_ray(self):
        import ray

        ray.init(ignore_reinit_error=True, num_gpus=0)
        yield
        ray.shutdown()

    def test_ray_eval_worker_basic(self):
        """Test submit/poll cycle with a simple environment."""

        def make_env():
            return TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(10))

        def make_policy(env):
            action_dim = env.action_spec.shape[-1]
            obs_dim = env.observation_spec["observation"].shape[-1]
            return TensorDictModule(
                nn.Linear(obs_dim, action_dim),
                in_keys=["observation"],
                out_keys=["action"],
            )

        worker = RayEvalWorker(
            init_fn=None,
            env_maker=make_env,
            policy_maker=make_policy,
            num_gpus=0,
        )
        try:
            # Before submit, poll returns None
            assert worker.poll() is None

            weights = (
                TensorDict.from_module(make_policy(make_env())).data.detach().cpu()
            )
            worker.submit(weights, max_steps=5)

            # Wait for result (blocking poll)
            result = worker.poll(timeout=30)
            assert result is not None
            assert "reward" in result
            assert "frames" in result
        finally:
            worker.shutdown()

    def test_ray_eval_worker_from_name(self):
        """Test that from_name can reconnect to a named actor."""

        def make_env():
            return TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(10))

        def make_policy(env):
            action_dim = env.action_spec.shape[-1]
            obs_dim = env.observation_spec["observation"].shape[-1]
            return TensorDictModule(
                nn.Linear(obs_dim, action_dim),
                in_keys=["observation"],
                out_keys=["action"],
            )

        worker = RayEvalWorker(
            init_fn=None,
            env_maker=make_env,
            policy_maker=make_policy,
            num_gpus=0,
            name="test_eval_worker",
        )
        try:
            # Reconnect via from_name
            worker2 = RayEvalWorker.from_name("test_eval_worker")

            weights = (
                TensorDict.from_module(make_policy(make_env())).data.detach().cpu()
            )
            # Submit through the reconnected handle
            worker2.submit(weights, max_steps=5)

            result = worker2.poll(timeout=30)
            assert result is not None
            assert "reward" in result
        finally:
            worker.shutdown()


@pytest.mark.skipif(not _has_procgen, reason="Procgen not found")
class TestProcgen:
    @pytest.mark.parametrize("envname", ["coinrun", "starpilot"])
    def test_procgen_envs_available(self, envname):
        # availability check
        assert envname in ProcgenEnv.available_envs

    def test_procgen_invalid_env_raises(self):
        with pytest.raises(ValueError):
            ProcgenEnv("this_env_does_not_exist")

    def test_procgen_num_envs_batch_size(self):
        env = ProcgenEnv("coinrun", num_envs=3)
        try:
            td = env.reset()
            assert td["observation"].shape[0] == 3
        finally:
            env.close()

    def test_procgen_seeding_is_deterministic(self):
        # Procgen must be seeded at construction time via the seed parameter
        e1 = ProcgenEnv("coinrun", num_envs=2, seed=0)
        e2 = ProcgenEnv("coinrun", num_envs=2, seed=0)
        try:
            t1 = e1.reset()
            t2 = e2.reset()
            assert torch.equal(t1["observation"], t2["observation"])
        finally:
            e1.close()
            e2.close()

    def test_procgen_step_keys_and_shapes(self):
        env = ProcgenEnv("coinrun", num_envs=2)
        try:
            env.reset()
            td = env.rand_step()
            # After step, observation/reward/done are in td["next"]
            for k in ("observation", "reward", "done"):
                assert k in td["next"]
            assert td["next"]["observation"].shape[0] == 2
        finally:
            env.close()

    def test_procgen_env_creation_and_reset(self):
        env = ProcgenEnv("coinrun", num_envs=2)
        try:
            td = env.reset()
            assert td["observation"].shape[0] == 2
        finally:
            env.close()

    def test_procgen_check_env_specs(self):
        env = ProcgenEnv("coinrun", num_envs=2)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_procgen_wrapper(self):
        import procgen as procgen_lib

        raw_env = procgen_lib.ProcgenEnv(num_envs=2, env_name="coinrun")
        env = ProcgenWrapper(env=raw_env)
        try:
            check_env_specs(env)
            td = env.reset()
            assert td["observation"].shape[0] == 2
            out = env.rand_step()
            # After step, observation/reward are in out["next"]
            assert "observation" in out["next"]
            assert "reward" in out["next"]
        finally:
            env.close()

    @pytest.mark.parametrize("distribution_mode", ["easy", "hard"])
    def test_procgen_distribution_mode(self, distribution_mode):
        env = ProcgenEnv("coinrun", num_envs=2, distribution_mode=distribution_mode)
        try:
            td = env.reset()
            assert td["observation"].shape[0] == 2
        finally:
            env.close()

    def test_procgen_start_level_num_levels(self):
        env = ProcgenEnv("coinrun", num_envs=2, start_level=0, num_levels=10)
        try:
            td = env.reset()
            assert td["observation"].shape[0] == 2
        finally:
            env.close()
