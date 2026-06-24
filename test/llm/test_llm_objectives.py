# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pytest
import tensordict
import torch

from tensordict import lazy_stack, MetaData, TensorDict
from torchrl._utils import logger
from torchrl.data import History, LazyStackStorage, ReplayBuffer
from torchrl.data.llm.history import _CHAT_TEMPLATES
from torchrl.envs.llm.transforms.kl import RetrieveLogProb
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from torchrl.modules.llm.policies.common import ChatHistory, Masks, Text, Tokens
from torchrl.objectives.llm.grpo import (
    CISPOLoss,
    CISPOLossOutput,
    GRPOLoss,
    GRPOLossOutput,
    MCAdvantage,
    MCAdvantageSelector,
)
from torchrl.objectives.llm.sft import SFTLoss

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None
prompts = [
    "Lorem ipsum dolor sit amet,",
    "consectetur adipiscing elit,",
    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation",
    "ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore",
    "eu fugiat nulla pariatur.",
]


@pytest.fixture(autouse=True, scope="module")
def set_list_to_stack():
    with tensordict.set_list_to_stack(True):
        yield


def _make_group_traj(group_id, rewards, group_key="group_id"):
    n_steps = len(rewards)
    return TensorDict(
        {
            group_key: torch.full((n_steps,), group_id),
            ("next", "reward"): torch.tensor(rewards).reshape(n_steps, 1),
            ("next", "done"): torch.tensor([False] * (n_steps - 1) + [True]).reshape(
                n_steps, 1
            ),
        },
        batch_size=[n_steps],
    )


class TestMCAdvantage:
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_mc_advantage(self, ndim):
        # make trajectories
        def make_silly_trajectory(n_steps=None):
            while True:
                if n_steps is None:
                    n_steps = torch.randint(low=2, high=100, size=(1,)).item()
                tds = []
                for _ in range(n_steps):
                    n_tokens = torch.randint(low=1, high=100, size=(1,)).item()
                    rewards = [torch.randn(n_tokens, 1)]
                    prompt = np.random.choice(prompts)
                    td = TensorDict(
                        query=prompt,  # MCAdvantage expects "query" key, not "text"
                        next=TensorDict(
                            reward=rewards, done=torch.zeros(1, dtype=torch.bool)
                        ),
                    )
                    tds.append(td)
                tds[-1]["next", "done"] = torch.ones(1, dtype=torch.bool)
                yield lazy_stack(tds)

        rb = ReplayBuffer(storage=LazyStackStorage(100))
        rb.append_transform(MCAdvantage(grpo_size=4))
        if ndim == 1:
            gen = make_silly_trajectory()
            for _ in range(100):
                trajectory = next(gen)
                rb.extend(trajectory)
            assert len(rb)
            s = rb.sample(1)
            assert "advantage" in s.keys()
        else:
            gen = make_silly_trajectory(n_steps=5)
            for _ in range(100):
                trajectory = lazy_stack([next(gen) for _ in range(3)])
                trajectory = trajectory.view(-1)
                rb.extend(trajectory)
            assert len(rb)
            s = rb.sample(1)
            assert "advantage" in s.keys()

    @pytest.mark.parametrize("agg", ["sum", "max", "mean"])
    def test_mc_advantage_trajectory_return(self, agg):
        # trajectory-level semantics: returns normalized across the group,
        # advantage broadcast to every step of each trajectory
        adv_t = MCAdvantage(grpo_size=3, prompt_key="group_id", trajectory_return=agg)
        trajs = [
            _make_group_traj(0, [0.0, 0.0]),
            _make_group_traj(0, [0.0, 1.0, 1.0]),
            _make_group_traj(0, [1.0, 0.0]),
        ]
        assert adv_t.inv(trajs[0]) is None
        assert adv_t.inv(trajs[1]) is None
        out = adv_t.inv(trajs[2])
        reduce = {"sum": torch.sum, "max": torch.max, "mean": torch.mean}[agg]
        returns = torch.stack([reduce(traj["next", "reward"]) for traj in trajs])
        expected = (returns - returns.mean()) / returns.std().clamp_min(1e-6)
        assert out["advantage"].shape == out["next", "reward"].shape
        for chunk, exp in zip(out["advantage"].squeeze(-1).split([2, 3, 2]), expected):
            torch.testing.assert_close(chunk, exp.expand_as(chunk))

    @pytest.mark.parametrize("group_key", ["group_id", ("meta", "group_id")])
    def test_mc_advantage_tensor_group_key(self, group_key):
        # tensor group identifiers (under a flat or nested key) are grouped by
        # value; interleaved groups complete independently
        adv_t = MCAdvantage(grpo_size=2, prompt_key=group_key, trajectory_return="sum")
        assert adv_t.inv(_make_group_traj(0, [1.0, 0.0], group_key=group_key)) is None
        assert adv_t.inv(_make_group_traj(1, [0.0, 0.0], group_key=group_key)) is None
        out0 = adv_t.inv(_make_group_traj(0, [0.0, 0.0], group_key=group_key))
        assert out0 is not None
        assert (out0[group_key] == 0).all()
        out1 = adv_t.inv(_make_group_traj(1, [0.0, 1.0], group_key=group_key))
        assert out1 is not None
        assert (out1[group_key] == 1).all()

    def test_mc_advantage_dynamic_sampling(self):
        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            keep_return_bounds=(0.1, 0.9),
        )
        # all-failed group: dropped
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        # all-succeeded group: dropped
        assert adv_t.inv(_make_group_traj(1, [1.0])) is None
        assert adv_t.inv(_make_group_traj(1, [1.0])) is None
        # mixed group: kept
        assert adv_t.inv(_make_group_traj(2, [0.0])) is None
        out = adv_t.inv(_make_group_traj(2, [1.0]))
        assert out is not None
        # processed groups (kept or dropped) do not linger in memory
        assert not adv_t.queues
        assert adv_t.completed_trajectories == 6
        assert adv_t.completed_decisions == 6
        assert adv_t.successful_trajectories == 3
        assert adv_t.trajectory_return_sum == 3.0
        assert adv_t.trajectory_return_max == 1.0
        assert adv_t.completed_groups == 3
        assert adv_t.written_groups == 1
        assert adv_t.dropped_groups == 2
        assert adv_t.queued_groups == 0
        assert adv_t.queued_trajectories == 0
        assert adv_t.max_queued_trajectories_per_group == 0
        adv_t.reset_stats()
        assert adv_t.completed_groups == 0
        assert adv_t.written_groups == 0
        assert adv_t.dropped_groups == 0
        assert adv_t.rescued_groups == 0
        assert adv_t.selected_trajectories == 0
        assert adv_t.unselected_trajectories == 0
        assert adv_t.completed_trajectories == 0
        assert adv_t.completed_decisions == 0

    def test_mc_advantage_candidate_selection_rescues_dynamic_sampling_group(self):
        selector = MCAdvantageSelector()
        assert selector.select(
            TensorDict({"return": torch.tensor([0.0, 0.0, 0.0, 1.0])}, [4]),
            group_size=2,
            keep_return_bounds=(0.1, 0.9),
        ) == [0, 3]
        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            keep_return_bounds=(0.1, 0.9),
            candidate_group_size=4,
            candidate_selector=selector,
        )
        for _ in range(3):
            assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        out = adv_t.inv(_make_group_traj(0, [1.0]))
        assert out is not None
        assert out.shape == (2,)
        assert out["advantage"].shape == out["next", "reward"].shape
        assert adv_t.completed_groups == 1
        assert adv_t.written_groups == 1
        assert adv_t.dropped_groups == 0
        assert adv_t.rescued_groups == 1
        assert adv_t.selected_trajectories == 2
        assert adv_t.unselected_trajectories == 2

    def test_mc_advantage_candidate_selector_reads_tensordict(self):
        class ScoreSelector(MCAdvantageSelector):
            def __init__(self):
                super().__init__(in_keys=[("trajectories", "selector_score")])

            def select(self, candidates, *, group_size, keep_return_bounds=None):
                trajectories = candidates.select(*self.in_keys, strict=True).get(
                    "trajectories"
                )
                scores = torch.stack(
                    [
                        trajectory.get("selector_score")[0]
                        for trajectory in trajectories.unbind(0)
                    ]
                )
                return scores.topk(group_size).indices.tolist()

        def make_scored_traj(score, rewards):
            traj = _make_group_traj(0, rewards)
            traj.set("selector_score", torch.full((len(rewards),), float(score)))
            return traj

        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            candidate_group_size=3,
            candidate_selection_min_size=3,
            candidate_selector=ScoreSelector(),
        )
        assert adv_t.inv(make_scored_traj(0, [0.0])) is None
        assert adv_t.inv(make_scored_traj(2, [1.0])) is None
        out = adv_t.inv(make_scored_traj(1, [0.0]))
        assert out is not None
        assert out["selector_score"].tolist() == [2.0, 1.0]

    def test_mc_advantage_candidate_selection_writes_as_soon_as_useful(self):
        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            keep_return_bounds=(0.1, 0.9),
            candidate_group_size=4,
        )
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        out = adv_t.inv(_make_group_traj(0, [1.0]))
        assert out is not None
        assert out.shape == (2,)
        assert adv_t.completed_groups == 1
        assert adv_t.written_groups == 1
        assert adv_t.dropped_groups == 0
        assert adv_t.queued_groups == 0
        assert adv_t.selected_trajectories == 2
        assert adv_t.unselected_trajectories == 0

    def test_mc_advantage_candidate_selection_rescues_before_max_candidates(self):
        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            keep_return_bounds=(0.1, 0.9),
            candidate_group_size=4,
        )
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        out = adv_t.inv(_make_group_traj(0, [1.0]))
        assert out is not None
        assert out.shape == (2,)
        assert adv_t.completed_groups == 1
        assert adv_t.written_groups == 1
        assert adv_t.rescued_groups == 1
        assert adv_t.selected_trajectories == 2
        assert adv_t.unselected_trajectories == 1

    def test_mc_advantage_candidate_selection_drops_without_valid_subset(self):
        adv_t = MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
            keep_return_bounds=(0.1, 0.9),
            candidate_group_size=4,
        )
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.completed_groups == 0
        assert adv_t.queued_trajectories == 2
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.completed_groups == 0
        assert adv_t.queued_trajectories == 3
        assert adv_t.inv(_make_group_traj(0, [0.0])) is None
        assert adv_t.completed_groups == 1
        assert adv_t.written_groups == 0
        assert adv_t.dropped_groups == 1

    def test_mc_advantage_trajectory_return_rb(self):
        # write path through a replay buffer: complete mixed groups are written,
        # degenerate groups are filtered out
        rb = ReplayBuffer(storage=LazyStackStorage(100))
        rb.append_transform(
            MCAdvantage(
                grpo_size=2,
                prompt_key="group_id",
                trajectory_return="sum",
                keep_return_bounds=(0.1, 0.9),
            )
        )
        rb.extend(_make_group_traj(0, [0.0, 0.0]))
        assert len(rb) == 0
        rb.extend(_make_group_traj(0, [0.0, 1.0]))
        assert len(rb) == 4
        rb.extend(_make_group_traj(1, [0.0, 0.0]))
        rb.extend(_make_group_traj(1, [0.0, 0.0]))
        assert len(rb) == 4
        sample = rb.sample(4)
        assert "advantage" in sample.keys()

    def test_mc_advantage_multi_done_flat_batch(self):
        # a single inv call carrying several concatenated trajectories (the
        # layout a collector yields with trajs_per_batch + traj_format="cat") is
        # split on the done flags and processed per trajectory
        adv_t = MCAdvantage(grpo_size=2, prompt_key="group_id", trajectory_return="sum")
        flat = torch.cat(
            [
                _make_group_traj(0, [1.0, 0.0]),
                _make_group_traj(0, [0.0, 0.0, 0.0]),
                _make_group_traj(1, [0.0]),
            ],
            0,
        )
        out = adv_t.inv(flat)
        # group 0 completed (both trajectories come out), group 1 stays queued
        assert out.shape == (5,)
        returns = torch.tensor([1.0, 0.0])
        expected = (returns - returns.mean()) / returns.std().clamp_min(1e-6)
        torch.testing.assert_close(
            out["advantage"].squeeze(-1),
            torch.cat([expected[0].expand(2), expected[1].expand(3)]),
        )
        assert sum(len(q) for q in adv_t.queues.values()) == 1
        assert adv_t.max_queued_trajectories_per_group == 1

    def test_mc_advantage_validation(self):
        with pytest.raises(ValueError, match="trajectory_return must be one of"):
            MCAdvantage(grpo_size=2, trajectory_return="prod")
        with pytest.raises(ValueError, match="dynamic sampling"):
            MCAdvantage(grpo_size=2, keep_return_bounds=(0.1, 0.9))
        with pytest.raises(ValueError, match="increasing"):
            MCAdvantage(
                grpo_size=2, trajectory_return="sum", keep_return_bounds=(0.9, 0.1)
            )
        # group-relative normalization over a single trajectory would silently
        # produce NaN advantages (std of one element)
        with pytest.raises(ValueError, match="grpo_size >= 2"):
            MCAdvantage(grpo_size=1, trajectory_return="sum")
        with pytest.raises(ValueError, match="candidate_group_size"):
            MCAdvantage(grpo_size=2, trajectory_return="sum", candidate_group_size=1)
        with pytest.raises(ValueError, match="candidate_group_size > grpo_size"):
            MCAdvantage(grpo_size=2, candidate_group_size=4)
        with pytest.raises(ValueError, match="candidate_selection_min_size"):
            MCAdvantage(
                grpo_size=2,
                trajectory_return="sum",
                candidate_group_size=4,
                candidate_selection_min_size=1,
            )
        with pytest.raises(ValueError, match="candidate_selection_min_size"):
            MCAdvantage(
                grpo_size=2,
                trajectory_return="sum",
                candidate_group_size=4,
                candidate_selection_min_size=5,
            )
        with pytest.raises(ValueError, match="strategy must be one of"):
            MCAdvantageSelector("unknown")


# Mock infrastructure moved to conftest.py


def _mock_data_grpo(vocab_size: int, device: torch.device | str = "cpu") -> TensorDict:
    from transformers import AutoTokenizer

    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    prompt = History(
        role=["system", "user"],
        content=["You are a useful assistant.", "What is 2+2?"],
        batch_size=(2,),
        device=device,
    )
    response = History(
        role=["assistant"],
        content=["2 + 2 = 4."],
        batch_size=(1,),
        device=device,
    )
    full_history = prompt.extend(response, inplace=False)
    history = ChatHistory(
        prompt=prompt,
        response=response,
        full=full_history,
        device=device,
    )
    batch_size = 1

    # Expand history to match batch size before getting tokens
    history = history.expand((batch_size,))
    next_history = ChatHistory(
        prompt=full_history,
        device=device,
    )
    next_history = next_history.expand((batch_size,))

    # Now get tokens from the expanded history objects
    tokens_full = history.to_tokens(tokenizer)
    next_tokens = next_history.to_tokens(tokenizer)

    # Get the actual sequence length from the tokens
    # tokens_full has structure with "full" key containing the actual tokens
    # We need to get the padded version to know the actual length
    tokens_input_ids = tokens_full.get(
        "full", as_padded_tensor=True, padding_side="left", padding_value=0
    )
    seq_len = tokens_input_ids.shape[-1]

    # Create tensors with proper shapes
    reward = torch.randn(batch_size, seq_len, 1, device=device)
    done = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device)
    advantage = torch.randn(batch_size, seq_len, 1, device=device)
    log_probs = torch.randn_like(tokens_full, dtype=torch.float32, device=device)

    # Create attention mask (all ones for non-padded tokens)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Import Masks to create proper mask structure
    masks = Masks(
        all_attention_mask=attention_mask,
        all_assistant_mask=None,  # Will be computed by the wrapper
        padded=MetaData(True),
        device=device,
    )

    data = TensorDict(
        {
            "advantage": advantage,
            "history": history,
            "tokens": tokens_full % vocab_size,
            "masks": masks,
            "next": {
                "history": next_history,
                "tokens": next_tokens % vocab_size,
                "reward": reward,
                "done": done,
            },
            "log_probs": log_probs,
        },
        batch_size=(batch_size,),
    )
    return data


class TestLosses:
    def test_grpo_token_mean_expands_token_mask(self):
        """Test token_mean aggregation with per-token values and masks."""
        loss_fn = GRPOLoss(actor_network=None, aggregation="token_mean")
        value = torch.arange(6, dtype=torch.float32).view(2, 3, 1)
        mask = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )

        result = loss_fn._aggregate_loss_value(value, mask)

        expected_mask = mask.unsqueeze(-1).expand_as(value)
        expected = value[expected_mask].mean()
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dapo", [True, False], ids=["dapo", "symmetric"])
    def test_grpo(self, mock_transformer_model, dapo):
        """Test GRPO loss computation with mock models."""
        vocab_size = 1024
        device = torch.device("cpu")
        if dapo:
            eps_low = 0.20
            eps_high = 0.28
            eps = (eps_low, eps_high)
        else:
            eps = 0.20
        # Create mock model and wrap it
        model = mock_transformer_model(vocab_size=vocab_size, device=device)
        actor_network = TransformersWrapper(
            model,
            generate=False,
            pad_output=True,
            input_mode="history",
        )

        # Create loss module
        loss_fn = GRPOLoss(actor_network, clip_epsilon=eps)

        # Create fake data
        data = _mock_data_grpo(vocab_size=vocab_size, device=device)

        # Compute loss
        loss_vals = loss_fn(data)

        # Assertions: Check output type and structure

        assert isinstance(
            loss_vals, GRPOLossOutput
        ), f"Expected GRPOLossOutput, got {type(loss_vals)}"

        # Check that all expected keys are present
        assert hasattr(loss_vals, "loss_objective"), "Missing loss_objective"
        assert hasattr(loss_vals, "clip_fraction"), "Missing clip_fraction"
        assert hasattr(loss_vals, "kl_approx"), "Missing kl_approx"
        assert hasattr(loss_vals, "ESS"), "Missing ESS"
        assert hasattr(loss_vals, "entropy"), "Missing entropy"
        assert hasattr(loss_vals, "loss_entropy"), "Missing loss_entropy"

        # Check tensor shapes (all losses should be scalars after reduction)
        assert (
            loss_vals.loss_objective.shape == ()
        ), f"loss_objective should be scalar, got {loss_vals.loss_objective.shape}"
        assert (
            loss_vals.clip_fraction.shape == ()
        ), f"clip_fraction should be scalar, got {loss_vals.clip_fraction.shape}"
        assert (
            loss_vals.kl_approx.shape == ()
        ), f"kl_approx should be scalar, got {loss_vals.kl_approx.shape}"
        assert (
            loss_vals.ESS.shape == ()
        ), f"ESS should be scalar, got {loss_vals.ESS.shape}"

        # Check that losses are finite
        assert torch.isfinite(loss_vals.loss_objective), "loss_objective is not finite"
        assert torch.isfinite(loss_vals.ESS), "ESS is not finite"

        # Check that clip_fraction is in valid range [0, 1]
        assert (
            0 <= loss_vals.clip_fraction <= 1
        ), f"clip_fraction out of range: {loss_vals.clip_fraction}"

    def test_kl_mask_threshold(self, mock_transformer_model):
        """Test that kl_mask_threshold properly filters out high-KL tokens."""
        torch.manual_seed(42)
        vocab_size = 1024
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create mock model and wrap it
        model = mock_transformer_model(vocab_size=vocab_size, device=device)
        actor_network = TransformersWrapper(
            model,
            generate=False,
            pad_output=True,
            input_mode="history",
        )

        # Create fake data
        data = _mock_data_grpo(vocab_size=vocab_size, device=device)

        # First, test that the data works without any threshold
        loss_fn_baseline = GRPOLoss(
            actor_network, clip_epsilon=0.2, kl_mask_threshold=None
        )

        data_baseline = data.clone()
        loss_baseline = loss_fn_baseline(data_baseline)
        logger.info(f"Baseline loss (no threshold): {loss_baseline.loss_objective}")
        logger.info(f"Baseline ESS: {loss_baseline.ESS}")

        # Check baseline is valid
        if not torch.isfinite(loss_baseline.loss_objective):
            raise ValueError(
                f"Baseline loss is not finite: {loss_baseline.loss_objective}, skipping test"
            )

        # Now test with kl_mask_threshold enabled
        # Use a very high threshold that should not mask any tokens
        kl_threshold = 100.0  # Extremely high threshold to ensure no masking
        loss_fn_with_threshold = GRPOLoss(
            actor_network, clip_epsilon=0.2, kl_mask_threshold=kl_threshold
        )

        data_with_threshold = data.clone()
        loss_with_threshold = loss_fn_with_threshold(data_with_threshold)

        # Should produce valid output
        assert isinstance(loss_with_threshold, GRPOLossOutput)

        # Check that the loss is finite (with such a high threshold, it should be)
        assert torch.isfinite(
            loss_with_threshold.loss_objective
        ), f"loss_with_threshold is not finite: {loss_with_threshold.loss_objective}"
        assert torch.isfinite(
            loss_with_threshold.ESS
        ), f"ESS with threshold is not finite: {loss_with_threshold.ESS}"

        logger.info(
            f"Loss with high threshold (100.0): {loss_with_threshold.loss_objective}"
        )
        logger.info(f"ESS with high threshold: {loss_with_threshold.ESS}")

        # The losses should be identical or very similar since we're not masking anything
        # (the difference comes only from numerical precision)
        assert torch.isclose(
            loss_baseline.loss_objective, loss_with_threshold.loss_objective, rtol=1e-3
        ), f"Losses differ too much with high threshold: {loss_baseline.loss_objective} vs {loss_with_threshold.loss_objective}"

    def test_failure_missing_entries(self, mock_transformer_model):
        """Test that GRPO fails when required keys are missing but works without optional keys."""
        vocab_size = 1024
        device = torch.device("cpu")

        # Create mock model and wrap it
        model = mock_transformer_model(vocab_size=vocab_size, device=device)
        actor_network = TransformersWrapper(
            model,
            generate=False,
            pad_output=True,
            input_mode="history",
        )

        # Create loss module
        loss_fn = GRPOLoss(actor_network, clip_epsilon=0.2)

        # Create fake data
        data = _mock_data_grpo(vocab_size=vocab_size, device=device)

        # Test 1: Missing sample_log_prob (required) should fail
        data_missing_sample_log_prob = data.clone()
        data_missing_sample_log_prob.exclude(("log_probs", "full"), inplace=True)

        with pytest.raises(KeyError, match="Couldn't find the log-prob"):
            loss_fn(data_missing_sample_log_prob)

        # Test 2: Missing ref_log_probs (optional when kl_to_ref_coeff is None) should work
        data_missing_ref = data.clone()
        # Remove the ref_log_probs key if it exists
        if ("next", "ref_log_probs", "full") in data_missing_ref.keys(True):
            data_missing_ref.exclude(("next", "ref_log_probs", "full"), inplace=True)

        # Should work fine without ref_log_probs when kl_to_ref_coeff is None
        loss_vals = loss_fn(data_missing_ref)
        assert isinstance(loss_vals, GRPOLossOutput)
        assert torch.isfinite(loss_vals.loss_objective)

        # Test 3: Missing ref_log_probs when kl_to_ref_coeff is set should fail
        loss_fn_with_kl = GRPOLoss(actor_network, clip_epsilon=0.2, kl_to_ref_coeff=0.1)

        data_missing_ref_for_kl = data.clone()
        if ("next", "ref_log_probs", "full") in data_missing_ref_for_kl.keys(True):
            data_missing_ref_for_kl.exclude(
                ("next", "ref_log_probs", "full"), inplace=True
            )

        with pytest.raises(KeyError, match="Couldn't find the ref log-prob"):
            loss_fn_with_kl(data_missing_ref_for_kl)

    def test_cispo(self, mock_transformer_model):
        """Test CISPO loss computation with mock models."""
        vocab_size = 1024
        device = torch.device("cpu")
        eps = 0.20

        # Create mock model and wrap it
        model = mock_transformer_model(vocab_size=vocab_size, device=device)
        actor_network = TransformersWrapper(
            model,
            generate=False,
            pad_output=True,
            input_mode="history",
        )

        # Create loss module

        loss_fn = CISPOLoss(actor_network, clip_epsilon=eps)

        # Create fake data
        data = _mock_data_grpo(vocab_size=vocab_size, device=device)

        # Compute loss
        loss_vals = loss_fn(data)

        # Assertions: Check output type and structure

        assert isinstance(
            loss_vals, CISPOLossOutput
        ), f"Expected CISPOLossOutput, got {type(loss_vals)}"

        # Check that all expected keys are present (same as GRPO)
        assert hasattr(loss_vals, "loss_objective"), "Missing loss_objective"
        assert hasattr(loss_vals, "clip_fraction"), "Missing clip_fraction"
        assert hasattr(loss_vals, "kl_approx"), "Missing kl_approx"
        assert hasattr(loss_vals, "ESS"), "Missing ESS"
        assert hasattr(loss_vals, "entropy"), "Missing entropy"
        assert hasattr(loss_vals, "loss_entropy"), "Missing loss_entropy"

        # Check tensor shapes (all losses should be scalars after reduction)
        assert (
            loss_vals.loss_objective.shape == ()
        ), f"loss_objective should be scalar, got {loss_vals.loss_objective.shape}"
        assert (
            loss_vals.clip_fraction.shape == ()
        ), f"clip_fraction should be scalar, got {loss_vals.clip_fraction.shape}"
        assert (
            loss_vals.kl_approx.shape == ()
        ), f"kl_approx should be scalar, got {loss_vals.kl_approx.shape}"
        assert (
            loss_vals.ESS.shape == ()
        ), f"ESS should be scalar, got {loss_vals.ESS.shape}"

        # Check that losses are finite
        assert torch.isfinite(loss_vals.loss_objective), "loss_objective is not finite"
        assert torch.isfinite(loss_vals.ESS), "ESS is not finite"

        # Check that clip_fraction is in valid range [0, 1]
        assert (
            0 <= loss_vals.clip_fraction <= 1
        ), f"clip_fraction out of range: {loss_vals.clip_fraction}"


class TestSFT:
    @pytest.fixture(scope="class")
    def data(self):
        from transformers import AutoTokenizer

        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you very much!",
                },
            ],
        ]
        history = History.from_chats(chats)
        assert history.shape == (2, 3)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token
        text = history[:, :-1].apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        )
        full_text = history.apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        )
        text_response = [
            txt[len(txt_start) :] for txt, txt_start in zip(full_text, text)
        ]
        td = TensorDict(
            text=Text(prompt=text, response=text_response, full=full_text),
            history=ChatHistory(
                full=history, prompt=history[..., :-1], response=history[..., -1:]
            ),
            next=TensorDict(
                reward=torch.randn(2, 1),
                done=torch.zeros(2, dtype=torch.bool),
                history=ChatHistory(prompt=history),
            ),
            batch_size=(2,),
        )
        yield lazy_stack(list(td.unbind(0)))

    def tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.chat_template = _CHAT_TEMPLATES["qwen"]
        return tokenizer

    @pytest.fixture(scope="class")
    def policy_train(self):
        from transformers import OPTConfig, OPTForCausalLM

        tokenizer = self.tokenizer()
        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            chat_template_name="qwen",
            input_mode="history",
            pad_output=False,
        )

        return policy_train, tokenizer

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test SFT"
    )
    @pytest.mark.parametrize("loss_function", ["sft", "minor_sft"])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("normalize_by_seq_length", [True, False])
    @pytest.mark.parametrize("kl_to_ref_coeff", [None, 0.1])
    def test_sft(
        self,
        loss_function,
        reduction,
        normalize_by_seq_length,
        kl_to_ref_coeff,
        data,
        policy_train,
    ):
        policy_train, tokenizer = policy_train
        loss = SFTLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            reduction=reduction,
            normalize_by_seq_length=normalize_by_seq_length,
            kl_to_ref_coeff=kl_to_ref_coeff if loss_function != "minor_sft" else None,
            loss_function=loss_function,
            beta=0.1,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )

        td = data
        if kl_to_ref_coeff is not None or loss_function == "minor_sft":
            policy_ref = TransformersWrapper(
                policy_train.model,
                tokenizer=tokenizer,
                generate=False,
                return_log_probs=True,
                chat_template_name="qwen",
                input_mode="history",
                pad_output=False,
            )
            transform = RetrieveLogProb(
                policy_ref,
                assistant_only=True,
                tokenizer_kwargs={"chat_template_name": "qwen"},
                tokenizer=tokenizer,
                log_probs_full_key=("ref_log_probs", "full"),
            )
            with torch.no_grad():
                # Compute ref log-probs
                transform(td)
        loss_vals = loss(td)
        if kl_to_ref_coeff is not None and loss_function != "minor_sft":
            assert loss_vals.loss_kl_to_ref.shape == ()
            assert loss_vals.kl_to_ref.shape == ()
        if reduction == "mean":
            assert loss_vals.loss_sft.shape == ()
        elif reduction == "sum":
            assert loss_vals.loss_sft.shape == ()
        elif reduction == "none":
            assert loss_vals.loss_sft.shape == (2,)
        assert loss_vals.sum(reduce=True).shape == ()

    def test_sft_assistant_only(self, data):
        from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]

        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            chat_template_name="qwen",
        )
        policy_ref = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            return_log_probs=True,
            chat_template_name="qwen",
        )
        transform = RetrieveLogProb(
            policy_ref,
            assistant_only=True,
            tokenizer_kwargs={"chat_template_name": "qwen"},
            tokenizer=tokenizer,
            log_probs_full_key=("ref_log_probs", "full"),
        )
        td = transform(data)
        assert td is data
        loss = SFTLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            reduction="mean",
            normalize_by_seq_length=True,
            kl_to_ref_coeff=0.1,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )
        loss(td)


@pytest.mark.slow
@pytest.mark.integration
class TestGRPOLossIntegration:
    """Integration tests for GRPOLoss with real models (vLLM + transformers)."""

    @pytest.fixture(scope="class")
    def transformers_instance(self):
        """Create transformers model and tokenizer for testing."""
        if not _has_transformers:
            pytest.skip("transformers not available")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @pytest.fixture(scope="class")
    def vllm_instance(self):
        """Create vLLM model and tokenizer for testing."""
        if not _has_vllm:
            pytest.skip("vllm not available")

        import vllm.envs as envs
        from transformers import AutoTokenizer
        from vllm import LLM

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        try:
            model = LLM("Qwen/Qwen2.5-0.5B")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Failed to load vLLM model: {e}")

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.parametrize("masking_strategy", ["sft", "rlhf"])
    @pytest.mark.skip(
        reason="GRPOLoss shape mismatch between masking strategies - needs investigation"
    )
    def test_grpo_loss_with_real_models(
        self,
        vllm_instance,
        transformers_instance,
        masking_strategy,
    ):
        """Test GRPOLoss with vLLM generation and transformers loss computation."""
        model, tokenizer = transformers_instance
        vllm_model, vllm_tokenizer = vllm_instance

        # Create sample input based on masking strategy
        if masking_strategy == "sft":
            # Use tokens input mode for SFT
            text = [
                "Are you happy? Say yes or no.",
                "What is 2+2?",
            ]
            tokenized = tokenizer(
                text, return_tensors="pt", padding=True, padding_side="left"
            )
            input_data = {
                "tokens": Tokens(prompt=tokenized["input_ids"]),
                "masks": Masks(all_attention_mask=tokenized["attention_mask"]),
            }
            input_mode = "tokens"
        else:
            # Use history input mode for RLHF
            chats = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Are you happy? Say yes or no."},
                ],
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            ]
            sample_history = History.from_chats(chats)
            input_data = {"history": ChatHistory(prompt=sample_history)}
            input_mode = "history"

        # Generate responses with vLLM
        wrapper_gen = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            generate=True,
            return_log_probs=True,
            pad_output=True,
            generate_kwargs={"max_tokens": 10},
        )

        td = TensorDict(input_data, batch_size=(2,)).to_lazystack(0)
        td = wrapper_gen(td)
        td["advantage"] = torch.randn(2, 1, 1)

        # Compute loss with transformers
        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            return_log_probs=True,
            pad_output=True,
        )

        loss_fn = GRPOLoss(actor_network=wrapper, masking_strategy=masking_strategy)

        # Should successfully compute loss
        result = loss_fn(td)
        assert result is not None
        assert hasattr(result, "loss_objective")
        assert torch.isfinite(result.loss_objective)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
