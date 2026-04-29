# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from functools import partial

import pytest
import torch
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.libs.meltingpot import MeltingpotEnv, MeltingpotWrapper
from torchrl.envs.libs.pettingzoo import _has_pettingzoo, PettingZooEnv
from torchrl.envs.libs.smacv2 import _has_smacv2, SMACv2Env
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.envs.utils import check_env_specs, MarlGroupMapType
from torchrl.modules import MaskedCategorical

_has_meltingpot = importlib.util.find_spec("meltingpot") is not None


@pytest.mark.skipif(not _has_pettingzoo, reason="PettingZoo not found")
class TestPettingZoo:
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize("use_mask", [True])
    @pytest.mark.parametrize("return_state", [True, False])
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ALL_IN_ONE_GROUP, MarlGroupMapType.ONE_GROUP_PER_AGENT],
    )
    def test_pistonball(
        self, parallel, continuous_actions, use_mask, return_state, group_map
    ):
        kwargs = {"n_pistons": 21, "continuous": continuous_actions}

        env = PettingZooEnv(
            task="pistonball_v6",
            parallel=parallel,
            seed=0,
            return_state=return_state,
            use_mask=use_mask,
            group_map=group_map,
            **kwargs,
        )

        check_env_specs(env)

    def test_dead_agents_done(self, seed=0):
        scenario_args = {"n_walkers": 3, "terminate_on_fall": False}

        env = PettingZooEnv(
            task="multiwalker_v9",
            parallel=True,
            seed=seed,
            use_mask=False,
            done_on_any=False,
            **scenario_args,
        )
        td_reset = env.reset(seed=seed)
        with pytest.raises(
            ValueError,
            match="Dead agents found in the environment, "
            "you need to set use_mask=True to allow this.",
        ):
            env.rollout(
                max_steps=500,
                break_when_any_done=True,  # This looks at root done set with done_on_any
                auto_reset=False,
                tensordict=td_reset,
            )

        for done_on_any in [True, False]:
            env = PettingZooEnv(
                task="multiwalker_v9",
                parallel=True,
                seed=seed,
                use_mask=True,
                done_on_any=done_on_any,
                **scenario_args,
            )
            td_reset = env.reset(seed=seed)
            td = env.rollout(
                max_steps=500,
                break_when_any_done=True,  # This looks at root done set with done_on_any
                auto_reset=False,
                tensordict=td_reset,
            )
            done = td.get(("next", "walker", "done"))
            mask = td.get(("next", "walker", "mask"))

            if done_on_any:
                assert not done[-1].all()  # Done triggered on any
            else:
                assert done[-1].all()  # Done triggered on all
            assert not done[
                mask
            ].any()  # When mask is true (alive agent), all agents are not done
            assert done[
                ~mask
            ].all()  # When mask is false (dead agent), all agents are done

    @pytest.mark.parametrize(
        "wins_player_0",
        [True, False],
    )
    def test_tic_tac_toe(self, wins_player_0):
        env = PettingZooEnv(
            task="tictactoe_v3",
            parallel=False,
            group_map={"player": ["player_1", "player_2"]},
            categorical_actions=False,
            seed=0,
            use_mask=True,
        )

        class Policy:
            action = 0
            t = 0

            def __call__(self, td):
                new_td = env.input_spec["full_action_spec"].zero()

                player_acting = 0 if self.t % 2 == 0 else 1
                other_player = 1 if self.t % 2 == 0 else 0
                # The acting player has "mask" True and "action_mask" set to the available actions
                assert td["player", "mask"][player_acting].all()
                assert td["player", "action_mask"][player_acting].any()
                # The non-acting player has "mask" False and "action_mask" set to all Trues
                assert not td["player", "mask"][other_player].any()
                assert td["player", "action_mask"][other_player].all()

                if self.t % 2 == 0:
                    if not wins_player_0 and self.t == 4:
                        new_td["player", "action"][0][self.action + 1] = 1
                    else:
                        new_td["player", "action"][0][self.action] = 1
                else:
                    new_td["player", "action"][1][self.action + 6] = 1
                if td["player", "mask"][1].all():
                    self.action += 1
                self.t += 1
                return td.update(new_td)

        td = env.rollout(100, policy=Policy())

        assert td.batch_size[0] == (5 if wins_player_0 else 6)
        assert (td[:-1]["next", "player", "reward"] == 0).all()
        if wins_player_0:
            assert (
                td[-1]["next", "player", "reward"] == torch.tensor([[1], [-1]])
            ).all()
        else:
            assert (
                td[-1]["next", "player", "reward"] == torch.tensor([[-1], [1]])
            ).all()

    @pytest.mark.parametrize("task", ["simple_v3"])
    def test_return_state(self, task):
        """Test return_state=True returns state properly from raw env.state().

        Regression test for handling PettingZoo state encoding.
        Verifies that the state returned in tensordict matches the raw state
        from env.state(), ensuring proper conversion of any state type.
        """
        env = PettingZooEnv(
            task=task,
            parallel=True,
            seed=0,
            use_mask=False,
            return_state=True,
        )
        check_env_specs(env)

        # Get raw environment to access .state() directly
        raw_env = env._env

        # Reset and check initial state
        td_reset = env.reset()
        assert "state" in td_reset.keys()

        # Get raw state from environment
        raw_state = raw_env.state()

        # Get encoded state from tensordict
        encoded_state = td_reset["state"]

        # Compare size, type, and values
        # Convert raw_state to tensor for comparison
        raw_state_tensor = torch.as_tensor(raw_state, dtype=torch.float32)

        # Verify shape matches
        assert encoded_state.shape == raw_state_tensor.shape, (
            f"Shape mismatch: tensordict state has shape {encoded_state.shape}, "
            f"but raw state has shape {raw_state_tensor.shape}"
        )

        # Verify dtype
        assert (
            encoded_state.dtype == torch.float32
        ), f"Encoded state dtype should be float32, got {encoded_state.dtype}"
        assert (
            str(raw_state.dtype) == "float32"
        ), f"Raw state dtype should be float32, got {raw_state.dtype}"

        # Verify values match
        assert torch.allclose(
            encoded_state, raw_state_tensor
        ), "State values mismatch: tensordict state differs from raw state"

        # Verify states are not all zeros
        r = env.rollout(10)
        assert (r["state"] != 0).any()
        assert (r["next", "state"] != 0).any()

    @pytest.mark.parametrize(
        "task",
        [
            "multiwalker_v9",
            "waterworld_v4",
            "pursuit_v4",
            "simple_spread_v3",
            "simple_v3",
            "rps_v2",
            "cooperative_pong_v5",
            "pistonball_v6",
        ],
    )
    def test_envs_one_group_parallel(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=True,
            seed=0,
            use_mask=False,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "multiwalker_v9",
            "waterworld_v4",
            "pursuit_v4",
            "simple_spread_v3",
            "simple_v3",
            "rps_v2",
            "cooperative_pong_v5",
            "pistonball_v6",
            "connect_four_v3",
            "tictactoe_v3",
            "chess_v6",
            "gin_rummy_v4",
            "tictactoe_v3",
        ],
    )
    def test_envs_one_group_aec(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=False,
            seed=0,
            use_mask=True,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "simple_adversary_v3",
            "simple_crypto_v3",
            "simple_push_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
            "simple_tag_v3",
            "simple_world_comm_v3",
            "knights_archers_zombies_v10",
            "basketball_pong_v3",
            "boxing_v2",
            "foozpong_v3",
        ],
    )
    def test_envs_more_groups_parallel(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=True,
            seed=0,
            use_mask=False,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "simple_adversary_v3",
            "simple_crypto_v3",
            "simple_push_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
            "simple_tag_v3",
            "simple_world_comm_v3",
            "knights_archers_zombies_v10",
            "basketball_pong_v3",
            "boxing_v2",
            "foozpong_v3",
            "go_v5",
        ],
    )
    def test_envs_more_groups_aec(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=False,
            seed=0,
            use_mask=True,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize("task", ["knights_archers_zombies_v10", "pistonball_v6"])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_vec_env(self, task, parallel, maybe_fork_ParallelEnv):
        env_fun = partial(
            PettingZooEnv,
            task=task,
            parallel=parallel,
            seed=0,
            use_mask=not parallel,
        )
        vec_env = maybe_fork_ParallelEnv(2, create_env_fn=env_fun)
        vec_env.rollout(100, break_when_any_done=False)

    def test_reset_parallel_env(self, maybe_fork_ParallelEnv):
        def base_env_fn():
            return PettingZooEnv(
                task="multiwalker_v9",
                parallel=True,
                seed=0,
                n_walkers=3,
                max_cycles=1000,
            )

        collector = Collector(
            lambda: maybe_fork_ParallelEnv(
                num_workers=2,
                create_env_fn=base_env_fn,
                device="cpu",
            ),
            policy=None,
            frames_per_batch=100,
            max_frames_per_traj=50,
            total_frames=200,
            reset_at_each_iter=False,
        )
        for _ in collector:
            pass
        collector.shutdown()

    @pytest.mark.parametrize("task", ["knights_archers_zombies_v10", "pistonball_v6"])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_collector(self, task, parallel):
        env_fun = partial(
            PettingZooEnv,
            task=task,
            parallel=parallel,
            seed=0,
            use_mask=not parallel,
        )
        collector = Collector(
            create_env_fn=env_fun, frames_per_batch=30, total_frames=60, policy=None
        )
        for _ in collector:
            break

    def test_single_agent_group_replay_buffer(self):
        """Regression test for gh#3515 - shape mismatch with single-agent group."""
        env = PettingZooEnv(
            task="simple_v3",
            parallel=True,
            seed=0,
            use_mask=False,
        )
        group = list(env.group_map.keys())[0]
        assert len(env.group_map[group]) == 1

        rollout = env.rollout(10)
        T = rollout.shape[0]
        n_agents = 1

        # Reshape to (1, T, n_agents) to reproduce the scenario from gh#3515
        # where a replay buffer Transform reshapes collector output to
        # (n_envs, traj_len, n_agents). When n_agents=1 the trailing dim of 1
        # caused _set_index_in_td to match the wrong number of batch dims.
        td = rollout.unsqueeze(0).unsqueeze(-1)
        assert td.shape == torch.Size([1, T, n_agents])

        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10_000, ndim=3),
            batch_size=4,
        )
        rb.extend(td)


@pytest.mark.skipif(not _has_smacv2, reason="SMACv2 not found")
class TestSmacv2:
    def test_env_procedural(self):
        distribution_config = {
            "n_units": 5,
            "n_enemies": 6,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.5, 0.2, 0.3],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }
        env = SMACv2Env(
            map_name="10gen_terran",
            capability_config=distribution_config,
            seed=0,
        )
        check_env_specs(env, seed=None)
        env.close()

    @pytest.mark.parametrize("categorical_actions", [True, False])
    @pytest.mark.parametrize("map", ["MMM2", "3s_vs_5z"])
    def test_env(self, map: str, categorical_actions):
        env = SMACv2Env(
            map_name=map,
            categorical_actions=categorical_actions,
            seed=0,
        )
        check_env_specs(env, seed=None)
        env.close()

    def test_parallel_env(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(
                num_workers=2,
                create_env_fn=lambda: SMACv2Env(
                    map_name="3s_vs_5z",
                    seed=0,
                ),
            ),
            ActionMask(
                action_key=("agents", "action"), mask_key=("agents", "action_mask")
            ),
        )
        check_env_specs(env, seed=None)
        env.close()

    def test_collector(self):
        env = SMACv2Env(map_name="MMM2", seed=0, categorical_actions=True)
        in_feats = env.observation_spec["agents", "observation"].shape[-1]
        out_feats = env.full_action_spec[env.action_key].space.n

        module = TensorDictModule(
            nn.Linear(in_feats, out_feats),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        prob = ProbabilisticTensorDictModule(
            in_keys={"logits": ("agents", "logits"), "mask": ("agents", "action_mask")},
            out_keys=[("agents", "action")],
            distribution_class=MaskedCategorical,
        )
        actor = TensorDictSequential(module, prob)

        collector = Collector(env, policy=actor, frames_per_batch=20, total_frames=40)
        for _ in collector:
            break
        collector.shutdown()


@pytest.mark.skipif(not _has_meltingpot, reason="Meltingpot not found")
class TestMeltingpot:
    @pytest.mark.parametrize("substrate", MeltingpotWrapper.available_envs)
    def test_all_envs(self, substrate):
        env = MeltingpotEnv(substrate=substrate)
        check_env_specs(env)

    def test_passing_config(self, substrate="commons_harvest__open"):
        from meltingpot import substrate as mp_substrate

        substrate_config = mp_substrate.get_config(substrate)
        env_torchrl = MeltingpotEnv(substrate_config)
        env_torchrl.rollout(max_steps=5)

    def test_wrapper(self, substrate="commons_harvest__open"):
        from meltingpot import substrate as mp_substrate

        substrate_config = mp_substrate.get_config(substrate)
        mp_env = mp_substrate.build_from_config(
            substrate_config, roles=substrate_config.default_player_roles
        )
        env_torchrl = MeltingpotWrapper(env=mp_env)
        env_torchrl.rollout(max_steps=5)

    @pytest.mark.parametrize("max_steps", [1, 5])
    def test_max_steps(self, max_steps):
        env = MeltingpotEnv(substrate="commons_harvest__open", max_steps=max_steps)
        td = env.rollout(max_steps=100, break_when_any_done=True)
        assert td.batch_size[0] == max_steps

    @pytest.mark.parametrize("categorical_actions", [True, False])
    def test_categorical_actions(self, categorical_actions):
        env = MeltingpotEnv(
            substrate="commons_harvest__open", categorical_actions=categorical_actions
        )
        check_env_specs(env)

    @pytest.mark.parametrize("rollout_steps", [1, 3])
    def test_render(self, rollout_steps):
        env = MeltingpotEnv(substrate="commons_harvest__open")
        td = env.rollout(2)
        rollout_penultimate_image = td[-1].get("RGB")
        rollout_last_image = td[-1].get(("next", "RGB"))
        image_from_env = env.get_rgb_image()
        assert torch.equal(rollout_last_image, image_from_env)
        assert not torch.equal(rollout_penultimate_image, image_from_env)
