# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
from functools import partial

import numpy as np
import pytest
import torch

from _envs_common import _has_cairosvg, _has_chess, _has_transformers, _has_tv
from tensordict import assert_allclose_td, set_capture_non_tensor_stack, TensorDict
from tensordict.tensorclass import NonTensorStack

from test_model_based import TestModelBasedEnvBase

from torchrl import set_auto_unwrap_transformed_env
from torchrl.collectors import Collector
from torchrl.data.tensor_specs import Composite, NonTensor, Unbounded
from torchrl.envs import (
    AsyncEnvPool,
    ChessEnv,
    ConditionalSkip,
    EnvBase,
    ParallelEnv,
    SerialEnv,
)
from torchrl.envs.batched_envs import (
    _has_float64_leaf,
    _td_to_device_mps_safe,
    _to_device_mps_safe,
)
from torchrl.envs.transforms import StepCounter, TransformedEnv
from torchrl.envs.transforms.transforms import Tokenizer
from torchrl.envs.utils import check_env_specs
from torchrl.modules import RandomPolicy
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import (
    ContinuousActionConvMockEnv,
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    DiscreteActionConvMockEnv,
    DiscreteActionConvMockEnvNumpy,
    DiscreteActionVecMockEnv,
    DummyModelBasedEnvBase,
    EnvWithMetadata,
    HeterogeneousCountingEnv,
    HistoryTransform,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
    MockSerialEnv,
    MultiKeyCountingEnv,
    NestedCountingEnv,
    Str2StrEnv,
)


@pytest.mark.parametrize(
    "envclass",
    [
        EnvWithMetadata,
        ContinuousActionConvMockEnv,
        ContinuousActionConvMockEnvNumpy,
        ContinuousActionVecMockEnv,
        CountingBatchedEnv,
        CountingEnv,
        DiscreteActionConvMockEnv,
        DiscreteActionConvMockEnvNumpy,
        DiscreteActionVecMockEnv,
        partial(
            DummyModelBasedEnvBase, world_model=TestModelBasedEnvBase.world_model()
        ),
        MockBatchedLockedEnv,
        MockBatchedUnLockedEnv,
        MockSerialEnv,
        NestedCountingEnv,
        HeterogeneousCountingEnv,
        MultiKeyCountingEnv,
        Str2StrEnv,
    ],
)
def test_mocking_envs(envclass):
    with set_capture_non_tensor_stack(False):
        env = envclass()
        with pytest.warns(UserWarning, match="model based") if isinstance(
            env, DummyModelBasedEnvBase
        ) else contextlib.nullcontext():
            env.set_seed(100)
        reset = env.reset()
        _ = env.rand_step(reset)
        r = env.rollout(3)
        with pytest.warns(UserWarning, match="model based") if isinstance(
            env, DummyModelBasedEnvBase
        ) else contextlib.nullcontext():
            check_env_specs(env, seed=100, return_contiguous=False)


# fen strings for board positions generated with:
# https://lichess.org/editor
@pytest.mark.skipif(not _has_chess, reason="chess not found")
class TestChessEnv:
    @pytest.mark.parametrize("include_pgn", [False, True])
    @pytest.mark.parametrize("include_fen", [False, True])
    @pytest.mark.parametrize("stateful", [False, True])
    @pytest.mark.parametrize("include_hash", [False, True])
    @pytest.mark.parametrize("include_san", [False, True])
    def test_env(self, stateful, include_pgn, include_fen, include_hash, include_san):
        with pytest.raises(
            RuntimeError, match="At least one state representation"
        ) if not stateful and not include_pgn and not include_fen else contextlib.nullcontext():
            env = ChessEnv(
                stateful=stateful,
                include_pgn=include_pgn,
                include_fen=include_fen,
                include_hash=include_hash,
                include_san=include_san,
            )
            # Because we always use mask_actions=True
            assert isinstance(env, TransformedEnv)
            check_env_specs(env)
            if include_hash:
                if include_fen:
                    assert "fen_hash" in env.observation_spec.keys()
                if include_pgn:
                    assert "pgn_hash" in env.observation_spec.keys()
                if include_san:
                    assert "san_hash" in env.observation_spec.keys()

    # Test that `include_hash_inv=True` allows us to specify the board state
    # with just the "fen_hash" or "pgn_hash", not "fen" or "pgn", when taking a
    # step in the env.
    @pytest.mark.parametrize(
        "include_fen,include_pgn",
        [[True, False], [False, True]],
    )
    @pytest.mark.parametrize("stateful", [True, False])
    def test_env_hash_inv(self, include_fen, include_pgn, stateful):
        env = ChessEnv(
            include_fen=include_fen,
            include_pgn=include_pgn,
            include_hash=True,
            include_hash_inv=True,
            stateful=stateful,
        )
        env.check_env_specs()

        def exclude_fen_and_pgn(td):
            td = td.exclude("fen")
            td = td.exclude("pgn")
            return td

        td0 = env.reset()

        if include_fen:
            env_check_fen = ChessEnv(
                include_fen=True,
                stateful=stateful,
            )

        if include_pgn:
            env_check_pgn = ChessEnv(
                include_pgn=True,
                stateful=stateful,
            )

        for _ in range(8):
            td1 = env.rand_step(exclude_fen_and_pgn(td0.clone()))

            # Confirm that fen/pgn was not used to determine the board state
            assert "fen" not in td1.keys()
            assert "pgn" not in td1.keys()

            if include_fen:
                assert (td1["fen_hash"] == td0["fen_hash"]).all()
                assert "fen" in td1["next"]

                # Check that if we start in the same board state and perform the
                # same action in an env that does not use hashes, we obtain the
                # same next board state. This confirms that we really can
                # successfully specify the board state with a hash.
                td0_check = td1.clone().exclude("next").update({"fen": td0["fen"]})
                assert (
                    env_check_fen.step(td0_check)["next", "fen"] == td1["next", "fen"]
                )

            if include_pgn:
                assert (td1["pgn_hash"] == td0["pgn_hash"]).all()
                assert "pgn" in td1["next"]

                td0_check = td1.clone().exclude("next").update({"pgn": td0["pgn"]})
                assert (
                    env_check_pgn.step(td0_check)["next", "pgn"] == td1["next", "pgn"]
                )

            td0 = td1["next"]

    @pytest.mark.skipif(not _has_tv, reason="torchvision not found.")
    @pytest.mark.skipif(not _has_cairosvg, reason="cairosvg not found.")
    @pytest.mark.parametrize("stateful", [False, True])
    def test_chess_rendering(self, stateful):
        env = ChessEnv(stateful=stateful, include_fen=True, pixels=True)
        env.check_env_specs()
        r = env.rollout(3)
        assert "pixels" in r

    def test_pgn_bijectivity(self):
        np.random.seed(0)
        pgn = ChessEnv._PGN_RESTART
        board = ChessEnv._pgn_to_board(pgn)
        pgn_prev = pgn
        for _ in range(10):
            moves = list(board.legal_moves)
            move = np.random.choice(moves)
            board.push(move)
            pgn_move = ChessEnv._board_to_pgn(board)
            assert pgn_move != pgn_prev
            assert pgn_move == ChessEnv._board_to_pgn(ChessEnv._pgn_to_board(pgn_move))
            assert pgn_move == ChessEnv._add_move_to_pgn(pgn_prev, move)
            pgn_prev = pgn_move

    def test_consistency(self):
        env0_stateful = ChessEnv(stateful=True, include_pgn=True, include_fen=True)
        env1_stateful = ChessEnv(stateful=True, include_pgn=False, include_fen=True)
        env2_stateful = ChessEnv(stateful=True, include_pgn=True, include_fen=False)
        env0_stateless = ChessEnv(stateful=False, include_pgn=True, include_fen=True)
        env1_stateless = ChessEnv(stateful=False, include_pgn=False, include_fen=True)
        env2_stateless = ChessEnv(stateful=False, include_pgn=True, include_fen=False)
        torch.manual_seed(0)
        r1_stateless = env1_stateless.rollout(50, break_when_any_done=False)
        torch.manual_seed(0)
        r1_stateful = env1_stateful.rollout(50, break_when_any_done=False)
        torch.manual_seed(0)
        r2_stateless = env2_stateless.rollout(50, break_when_any_done=False)
        torch.manual_seed(0)
        r2_stateful = env2_stateful.rollout(50, break_when_any_done=False)
        torch.manual_seed(0)
        r0_stateless = env0_stateless.rollout(50, break_when_any_done=False)
        torch.manual_seed(0)
        r0_stateful = env0_stateful.rollout(50, break_when_any_done=False)
        assert (r0_stateless["action"] == r1_stateless["action"]).all()
        assert (r0_stateless["action"] == r2_stateless["action"]).all()
        assert (r0_stateless["action"] == r0_stateful["action"]).all()
        assert (r1_stateless["action"] == r1_stateful["action"]).all()
        assert (r2_stateless["action"] == r2_stateful["action"]).all()

    @pytest.mark.parametrize(
        "include_fen,include_pgn", [[True, False], [False, True], [True, True]]
    )
    @pytest.mark.parametrize("stateful", [False, True])
    def test_san(self, stateful, include_fen, include_pgn):
        torch.manual_seed(0)
        env = ChessEnv(
            stateful=stateful,
            include_pgn=include_pgn,
            include_fen=include_fen,
            include_san=True,
        )
        r = env.rollout(100, break_when_any_done=False)
        sans = r["next", "san"]
        actions = [env.san_moves.index(san) for san in sans]
        i = 0

        def policy(td):
            nonlocal i
            td["action"] = actions[i]
            i += 1
            return td

        r2 = env.rollout(100, policy=policy, break_when_any_done=False)
        assert_allclose_td(r, r2)

    @pytest.mark.parametrize(
        "include_fen,include_pgn", [[True, False], [False, True], [True, True]]
    )
    @pytest.mark.parametrize("stateful", [False, True])
    def test_rollout(self, stateful, include_pgn, include_fen):
        torch.manual_seed(0)
        env = ChessEnv(
            stateful=stateful, include_pgn=include_pgn, include_fen=include_fen
        )
        r = env.rollout(500, break_when_any_done=False)
        assert r.shape == (500,)

    @pytest.mark.parametrize(
        "include_fen,include_pgn", [[True, False], [False, True], [True, True]]
    )
    @pytest.mark.parametrize("stateful", [False, True])
    def test_reset_white_to_move(self, stateful, include_pgn, include_fen):
        env = ChessEnv(
            stateful=stateful, include_pgn=include_pgn, include_fen=include_fen
        )
        fen = "5k2/4r3/8/8/8/1Q6/2K5/8 w - - 0 1"
        td = env.reset(TensorDict({"fen": fen}))
        if include_fen:
            assert td["fen"] == fen
            assert env.board.fen() == fen
        assert td["turn"] == env.lib.WHITE
        assert not td["done"]

    @pytest.mark.parametrize("include_fen,include_pgn", [[True, False], [True, True]])
    @pytest.mark.parametrize("stateful", [False, True])
    def test_reset_black_to_move(self, stateful, include_pgn, include_fen):
        env = ChessEnv(
            stateful=stateful, include_pgn=include_pgn, include_fen=include_fen
        )
        fen = "5k2/4r3/8/8/8/1Q6/2K5/8 b - - 0 1"
        td = env.reset(TensorDict({"fen": fen}))
        assert td["fen"] == fen
        assert env.board.fen() == fen
        assert td["turn"] == env.lib.BLACK
        assert not td["done"]

    @pytest.mark.parametrize("include_fen,include_pgn", [[True, False], [True, True]])
    @pytest.mark.parametrize("stateful", [False, True])
    def test_reset_done_error(self, stateful, include_pgn, include_fen):
        env = ChessEnv(
            stateful=stateful, include_pgn=include_pgn, include_fen=include_fen
        )
        fen = "1R3k2/2R5/8/8/8/8/2K5/8 b - - 0 1"
        with pytest.raises(ValueError) as e_info:
            env.reset(TensorDict({"fen": fen}))

        assert "Cannot reset to a fen that is a gameover state" in str(e_info)

    @pytest.mark.parametrize("reset_without_fen", [False, True])
    @pytest.mark.parametrize(
        "endstate", ["white win", "black win", "stalemate", "50 move", "insufficient"]
    )
    @pytest.mark.parametrize("include_pgn", [False, True])
    @pytest.mark.parametrize("include_fen", [True])
    @pytest.mark.parametrize("stateful", [False, True])
    def test_reward(
        self, stateful, reset_without_fen, endstate, include_pgn, include_fen
    ):
        if stateful and reset_without_fen:
            # reset_without_fen is only used for stateless env
            return

        env = ChessEnv(
            stateful=stateful, include_pgn=include_pgn, include_fen=include_fen
        )

        if endstate == "white win":
            fen = "5k2/2R5/8/8/8/1R6/2K5/8 w - - 0 1"
            expected_turn = env.lib.WHITE
            move = "Rb8#"
            expected_reward = 1
            expected_done = True

        elif endstate == "black win":
            fen = "5k2/6r1/8/8/8/8/7r/1K6 b - - 0 1"
            expected_turn = env.lib.BLACK
            move = "Rg1#"
            expected_reward = 1
            expected_done = True

        elif endstate == "stalemate":
            fen = "5k2/6r1/8/8/8/8/7r/K7 b - - 0 1"
            expected_turn = env.lib.BLACK
            move = "Rb7"
            expected_reward = 0.5
            expected_done = True

        elif endstate == "insufficient":
            fen = "5k2/8/8/8/3r4/2K5/8/8 w - - 0 1"
            expected_turn = env.lib.WHITE
            move = "Kxd4"
            expected_reward = 0.5
            expected_done = True

        elif endstate == "50 move":
            fen = "5k2/8/1R6/8/6r1/2K5/8/8 b - - 99 123"
            expected_turn = env.lib.BLACK
            move = "Kf7"
            expected_reward = 0.5
            expected_done = True

        elif endstate == "not_done":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            expected_turn = env.lib.WHITE
            move = "e4"
            expected_reward = 0
            expected_done = False

        else:
            raise RuntimeError(f"endstate not supported: {endstate}")

        if reset_without_fen:
            td = TensorDict({"fen": fen})
        else:
            td = env.reset(TensorDict({"fen": fen}))
            assert td["turn"] == expected_turn

        td["action"] = env._san_moves.index(move)
        td = env.step(td)["next"]
        assert td["done"] == expected_done
        assert td["reward"] == expected_reward
        assert td["turn"] == (not expected_turn)

    @pytest.mark.skipif(not _has_transformers, reason="transformers required")
    def test_chess_tokenized(self):
        env = ChessEnv(include_fen=True, stateful=True, include_san=True)
        assert isinstance(env.observation_spec["fen"], NonTensor)
        env = env.append_transform(
            Tokenizer(in_keys=["fen"], out_keys=["fen_tokenized"])
        )
        assert isinstance(env.observation_spec["fen"], NonTensor)
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.transform.transform_input_spec(env.base_env.input_spec)
        r = env.rollout(10, return_contiguous=False)
        assert "fen_tokenized" in r
        assert "fen" in r
        assert "fen_tokenized" in r["next"]
        assert "fen" in r["next"]
        ftd = env.fake_tensordict()
        assert "fen_tokenized" in ftd
        assert "fen" in ftd
        assert "fen_tokenized" in ftd["next"]
        assert "fen" in ftd["next"]
        env.check_env_specs()

    @pytest.mark.parametrize("stateful", [False, True])
    @pytest.mark.parametrize("include_san", [False, True])
    def test_env_reset_with_hash(self, stateful, include_san):
        env = ChessEnv(
            include_fen=True,
            include_hash=True,
            include_hash_inv=True,
            stateful=stateful,
            include_san=include_san,
        )
        cases = [
            # (fen, num_legal_moves)
            ("5R1k/8/8/8/6R1/8/8/5K2 b - - 0 1", 1),
            ("8/8/2kq4/4K3/1R3Q2/8/8/8 w - - 0 1", 2),
            ("6R1/8/8/4rq2/3pPk2/5n2/8/2B1R2K b - e3 0 1", 2),
        ]
        for fen, num_legal_moves in cases:
            # Load the state by fen.
            td = env.reset(TensorDict({"fen": fen}))
            assert td["fen"] == fen
            assert td["action_mask"].sum() == num_legal_moves
            # Reset to initial state just to make sure that the next reset
            # actually changes the state.
            assert env.reset()["action_mask"].sum() == 20
            # Load the state by fen hash and make sure it gives the same output
            # as before.
            td_check = env.reset(td.select("fen_hash"))
            assert (td_check == td).all()

    @pytest.mark.parametrize("include_fen", [False, True])
    @pytest.mark.parametrize("include_pgn", [False, True])
    @pytest.mark.parametrize("stateful", [False, True])
    @pytest.mark.parametrize("mask_actions", [False, True])
    def test_all_actions(self, include_fen, include_pgn, stateful, mask_actions):
        if not stateful and not include_fen and not include_pgn:
            # pytest.skip("fen or pgn must be included if not stateful")
            return

        env = ChessEnv(
            include_fen=include_fen,
            include_pgn=include_pgn,
            stateful=stateful,
            mask_actions=mask_actions,
        )
        td = env.reset()

        if not mask_actions:
            with pytest.raises(RuntimeError, match="Cannot generate legal actions"):
                env.all_actions()
            return

        # Choose random actions from the output of `all_actions`
        for _ in range(100):
            if stateful:
                all_actions = env.all_actions()
            else:
                # Reset theinitial state first, just to make sure
                # `all_actions` knows how to get the board state from the input.
                env.reset()
                all_actions = env.all_actions(td.clone())

            # Choose some random actions and make sure they match exactly one of
            # the actions from `all_actions`. This part is not tested when
            # `mask_actions == False`, because `rand_action` can pick illegal
            # actions in that case.
            if mask_actions:
                # TODO: Something is wrong in `ChessEnv.rand_action` which makes
                # it fail to work properly for stateless mode. It doesn't know
                # how to correctly reset the board state to what is given in the
                # tensordict before picking an action. When this is fixed, we
                # can get rid of the two `reset`s below
                if not stateful:
                    env.reset(td.clone())
                td_act = td.clone()
                for _ in range(10):
                    rand_action = env.rand_action(td_act)
                    assert (rand_action["action"] == all_actions["action"]).sum() == 1
                if not stateful:
                    env.reset()

            action_idx = torch.randint(0, all_actions.shape[0], ()).item()
            chosen_action = all_actions[action_idx]
            td = env.step(td.update(chosen_action))["next"]

            if td["done"]:
                td = env.reset()


@pytest.mark.parametrize("device", [None, *get_default_devices()])
@pytest.mark.parametrize("env_device", [None, *get_default_devices()])
class TestPartialSteps:
    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_parallel_partial_steps(
        self, use_buffers, device, env_device, maybe_fork_ParallelEnv
    ):
        with torch.device(device) if device is not None else contextlib.nullcontext():
            penv = maybe_fork_ParallelEnv(
                4,
                lambda: CountingEnv(max_steps=10, start_val=2, device=env_device),
                use_buffers=use_buffers,
                device=device,
            )
            try:
                td = penv.reset()
                psteps = torch.zeros(4, dtype=torch.bool)
                psteps[[1, 3]] = True
                td.set("_step", psteps)

                td.set("action", penv.full_action_spec[penv.action_key].one())
                td = penv.step(td)
                assert_allclose_td(td[0].get("next"), td[0], intersection=True)
                assert (td[1].get("next") != 0).any()
                assert_allclose_td(td[2].get("next"), td[2], intersection=True)
                assert (td[3].get("next") != 0).any()
            finally:
                penv.close()
                del penv

    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_parallel_partial_step_and_maybe_reset(
        self, use_buffers, device, env_device, maybe_fork_ParallelEnv
    ):
        with torch.device(device) if device is not None else contextlib.nullcontext():
            penv = maybe_fork_ParallelEnv(
                4,
                lambda: CountingEnv(max_steps=10, start_val=2, device=env_device),
                use_buffers=use_buffers,
                device=device,
            )
            try:
                td = penv.reset()
                psteps = torch.zeros(4, dtype=torch.bool, device=td.get("done").device)
                psteps[[1, 3]] = True
                td.set("_step", psteps)

                td.set("action", penv.full_action_spec[penv.action_key].one())
                td, tdreset = penv.step_and_maybe_reset(td)
                assert_allclose_td(td[0].get("next"), td[0], intersection=True)
                assert (td[1].get("next") != 0).any()
                assert_allclose_td(td[2].get("next"), td[2], intersection=True)
                assert (td[3].get("next") != 0).any()
            finally:
                penv.close()
                del penv

    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_serial_partial_steps(self, use_buffers, device, env_device):
        with torch.device(device) if device is not None else contextlib.nullcontext():
            penv = SerialEnv(
                4,
                lambda: CountingEnv(max_steps=10, start_val=2, device=env_device),
                use_buffers=use_buffers,
                device=device,
            )
            try:
                td = penv.reset()
                psteps = torch.zeros(4, dtype=torch.bool)
                psteps[[1, 3]] = True
                td.set("_step", psteps)

                td.set("action", penv.full_action_spec[penv.action_key].one())
                td = penv.step(td)
                assert_allclose_td(td[0].get("next"), td[0], intersection=True)
                assert (td[1].get("next") != 0).any()
                assert_allclose_td(td[2].get("next"), td[2], intersection=True)
                assert (td[3].get("next") != 0).any()
            finally:
                penv.close()
                del penv

    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_serial_partial_step_and_maybe_reset(self, use_buffers, device, env_device):
        with torch.device(device) if device is not None else contextlib.nullcontext():
            penv = SerialEnv(
                4,
                lambda: CountingEnv(max_steps=10, start_val=2, device=env_device),
                use_buffers=use_buffers,
                device=device,
            )
            td = penv.reset()
            psteps = torch.zeros(4, dtype=torch.bool)
            psteps[[1, 3]] = True
            td.set("_step", psteps)

            td.set("action", penv.full_action_spec[penv.action_key].one())
            td = penv.step(td)
            assert_allclose_td(td[0].get("next"), td[0], intersection=True)
            assert (td[1].get("next") != 0).any()
            assert_allclose_td(td[2].get("next"), td[2], intersection=True)
            assert (td[3].get("next") != 0).any()


class TestEnvWithHistory:
    @pytest.fixture(autouse=True, scope="class")
    def set_capture(self):
        with set_capture_non_tensor_stack(False), set_auto_unwrap_transformed_env(
            False
        ):
            yield
        return

    def _make_env(self, device, max_steps=10):
        return CountingEnv(device=device, max_steps=max_steps).append_transform(
            HistoryTransform()
        )

    def _make_skipping_env(self, device, max_steps=10):
        env = self._make_env(device=device, max_steps=max_steps)
        # skip every 3 steps
        env = env.append_transform(
            ConditionalSkip(lambda td: ((td["step_count"] % 3) == 2))
        )
        env = TransformedEnv(env, StepCounter())
        return env

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_env_history_base(self, device):
        env = self._make_env(device)
        env.check_env_specs()

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_skipping_history_env(self, device):
        env = self._make_skipping_env(device)
        env.check_env_specs()
        r = env.rollout(100)

    @pytest.mark.parametrize("device_env", [None, "cpu"])
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("batch_cls", [SerialEnv, "parallel"])
    @pytest.mark.parametrize("consolidate", [False, True])
    def test_env_history_base_batched(
        self, device, device_env, batch_cls, maybe_fork_ParallelEnv, consolidate
    ):
        if batch_cls == "parallel":
            batch_cls = maybe_fork_ParallelEnv
        env = batch_cls(
            2,
            lambda: self._make_env(device_env),
            device=device,
            consolidate=consolidate,
        )
        try:
            assert not env._use_buffers
            env.check_env_specs(break_when_any_done="both")
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("device_env", [None, "cpu"])
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("batch_cls", [SerialEnv, "parallel"])
    @pytest.mark.parametrize("consolidate", [False, True])
    def test_skipping_history_env_batched(
        self, device, device_env, batch_cls, maybe_fork_ParallelEnv, consolidate
    ):
        if batch_cls == "parallel":
            batch_cls = maybe_fork_ParallelEnv
        env = batch_cls(
            2,
            lambda: self._make_skipping_env(device_env),
            device=device,
            consolidate=consolidate,
        )
        try:
            env.check_env_specs()
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("device_env", [None, "cpu"])
    @pytest.mark.parametrize("collector_cls", [Collector])
    def test_env_history_base_collector(self, device_env, collector_cls):
        env = self._make_env(device_env)
        collector = collector_cls(
            env, RandomPolicy(env.full_action_spec), total_frames=35, frames_per_batch=5
        )
        for d in collector:
            for i in range(d.shape[0] - 1):
                assert (
                    d[i + 1]["history"].content[0] == d[i]["next", "history"].content[0]
                )

    @pytest.mark.parametrize("device_env", [None, "cpu"])
    @pytest.mark.parametrize("collector_cls", [Collector])
    def test_skipping_history_env_collector(self, device_env, collector_cls):
        env = self._make_skipping_env(device_env, max_steps=10)
        collector = collector_cls(
            env,
            lambda td: td.update(env.full_action_spec.one()),
            total_frames=35,
            frames_per_batch=5,
        )
        length = None
        count = 1
        for d in collector:
            for k in range(1, 5):
                if len(d[k]["history"].content) == 2:
                    count = 1
                    continue
                if count % 3 == 2:
                    assert (
                        d[k]["next", "history"].content
                        == d[k - 1]["next", "history"].content
                    ), (d["next", "history"].content, k, count)
                else:
                    assert d[k]["next", "history"].content[-1] == str(
                        int(d[k - 1]["next", "history"].content[-1]) + 1
                    ), (d["next", "history"].content, k, count)
                count += 1
            count += 1


class TestAsyncEnvPool:
    def make_env(self, *, makers, backend):
        return AsyncEnvPool(makers, backend=backend)

    @pytest.fixture(scope="module")
    def make_envs(self):
        yield [
            partial(CountingEnv),
            partial(CountingEnv),
            partial(CountingEnv),
            partial(CountingEnv),
        ]

    @pytest.mark.parametrize("backend", ["multiprocessing", "threading"])
    def test_specs(self, backend, make_envs):
        env = self.make_env(makers=make_envs, backend=backend)
        assert env.batch_size == (4,)
        try:
            r = env.reset()
            assert r.shape == env.shape
            s = env.rand_step(r)
            assert s.shape == env.shape
            env.check_env_specs(break_when_any_done="both")
        finally:
            env._maybe_shutdown()

    @pytest.mark.parametrize("backend", ["multiprocessing", "threading"])
    @pytest.mark.parametrize("min_get", [None, 1, 2])
    @set_capture_non_tensor_stack(False)
    def test_async_reset_and_step(self, backend, make_envs, min_get):
        env = self.make_env(makers=make_envs, backend=backend)
        try:
            env.async_reset_send(
                TensorDict(
                    {env._env_idx_key: NonTensorStack(*range(env.batch_size.numel()))},
                    batch_size=env.batch_size,
                )
            )
            r = env.async_reset_recv(min_get=min_get)
            if min_get is not None:
                assert r.numel() >= min_get
            assert env._env_idx_key in r
            # take an action
            r.set("action", torch.ones(r.shape + (1,)))
            env.async_step_send(r)
            s = env.async_step_recv(min_get=min_get)
            if min_get is not None:
                assert s.numel() >= min_get
            assert env._env_idx_key in s
        finally:
            env._maybe_shutdown()

    @pytest.mark.parametrize("backend", ["multiprocessing", "threading"])
    def test_async_transformed(self, backend, make_envs):
        base_env = self.make_env(makers=make_envs, backend=backend)
        try:
            env = TransformedEnv(base_env, StepCounter())
            env.check_env_specs(break_when_any_done="both")
        finally:
            base_env._maybe_shutdown()


def _has_mps():
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        return torch.mps.is_available()
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


@pytest.mark.skipif(not _has_mps(), reason="MPS device not available")
class TestMPSDeviceCasting:
    def test_mps_does_not_support_float64(self):
        """Assert that MPS still doesn't support float64.

        If this test fails, MPS has gained float64 support and the downcasts
        can be removed.
        """
        with pytest.raises(TypeError, match="MPS framework doesn't support float64"):
            torch.ones(2, dtype=torch.float64, device="mps")

    def test_to_device_mps_safe_float64_downcast(self):
        t = torch.randn(4, dtype=torch.float64, device="cpu")
        result = _to_device_mps_safe(t, torch.device("mps"))
        assert result.device.type == "mps"
        assert result.dtype == torch.float32

    def test_td_to_device_mps_safe_downcasts_float64(self):
        td = TensorDict(
            {
                "obs": torch.randn(3, dtype=torch.float64),
                "flag": torch.ones(1, dtype=torch.bool),
            },
            batch_size=[],
        )
        result = _td_to_device_mps_safe(td, torch.device("mps"))
        assert result["obs"].device.type == "mps"
        assert result["obs"].dtype == torch.float32
        assert result["flag"].device.type == "mps"
        assert result["flag"].dtype == torch.bool

    def test_has_float64_leaf(self):
        from torchrl.data import Unbounded

        spec_f64 = Composite(
            obs=Unbounded(shape=(3,), dtype=torch.float64, device="cpu")
        )
        assert _has_float64_leaf(spec_f64) is True

        spec_mixed_dtype_with_f64 = Composite(
            obs32=Unbounded(shape=(3,), dtype=torch.float32, device="cpu"),
            obs64=Unbounded(shape=(3,), dtype=torch.float64, device="cpu"),
        )
        assert _has_float64_leaf(spec_mixed_dtype_with_f64) is True

        spec_f32 = Composite(
            obs=Unbounded(shape=(3,), dtype=torch.float32, device="cpu")
        )
        assert _has_float64_leaf(spec_f32) is False

        inner = Composite(obs=Unbounded(shape=(3,), dtype=torch.float64, device="cpu"))
        outer = Composite(next=inner)
        assert _has_float64_leaf(outer) is True

        assert _has_float64_leaf(Composite()) is False

    class _Float64ObsEnv(EnvBase):
        """Minimal env that produces float64 observations on CPU."""

        def __init__(self):
            super().__init__(device="cpu")
            self.observation_spec = Composite(
                observation=Unbounded(shape=(4,), dtype=torch.float64),
            )
            self.action_spec = Unbounded(shape=(2,))
            self.reward_spec = Unbounded(shape=(1,))

        def _reset(self, tensordict):
            return TensorDict(
                {
                    "observation": torch.randn(4, dtype=torch.float64),
                    "done": torch.zeros(1, dtype=torch.bool),
                    "terminated": torch.zeros(1, dtype=torch.bool),
                },
                batch_size=[],
            )

        def _step(self, tensordict):
            return TensorDict(
                {
                    "observation": torch.randn(4, dtype=torch.float64),
                    "reward": torch.zeros(1, dtype=torch.float64),
                    "done": torch.zeros(1, dtype=torch.bool),
                    "terminated": torch.zeros(1, dtype=torch.bool),
                },
                batch_size=[],
            )

        def _set_seed(self, seed):
            return seed

    _MPS_FLOAT64_WARNING = r"Sub-environments produce float64 data but the batched env device is 'mps.*' which does not support float64\. All float64 specs and tensors will be downcast to float32\."

    def test_serial_env_mps_parent_cpu_worker_reset(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = SerialEnv(2, self._Float64ObsEnv, device="mps")
        try:
            td = env.reset()
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_serial_env_mps_parent_cpu_worker_rollout(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = SerialEnv(2, self._Float64ObsEnv, device="mps")
        try:
            policy = RandomPolicy(env.action_spec)
            td = env.rollout(max_steps=3, policy=policy)
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_env_mps_parent_cpu_worker_reset(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = ParallelEnv(2, self._Float64ObsEnv, device="mps")
        try:
            td = env.reset()
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_env_mps_parent_cpu_worker_rollout(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = ParallelEnv(2, self._Float64ObsEnv, device="mps")
        try:
            policy = RandomPolicy(env.action_spec)
            td = env.rollout(max_steps=3, policy=policy)
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_serial_env_no_buffers_mps_reset(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = SerialEnv(2, self._Float64ObsEnv, device="mps", use_buffers=False)
        try:
            td = env.reset()
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_serial_env_no_buffers_mps_rollout(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = SerialEnv(2, self._Float64ObsEnv, device="mps", use_buffers=False)
        try:
            policy = RandomPolicy(env.action_spec)
            td = env.rollout(max_steps=3, policy=policy)
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_env_no_buffers_mps_reset(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = ParallelEnv(2, self._Float64ObsEnv, device="mps", use_buffers=False)
        try:
            td = env.reset()
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_env_no_buffers_mps_rollout(self):
        with pytest.warns(UserWarning, match=self._MPS_FLOAT64_WARNING):
            env = ParallelEnv(2, self._Float64ObsEnv, device="mps", use_buffers=False)
        try:
            policy = RandomPolicy(env.action_spec)
            td = env.rollout(max_steps=3, policy=policy)
            assert td.device.type == "mps"
            assert td["observation"].dtype == torch.float32
        finally:
            env.close(raise_if_closed=False)
