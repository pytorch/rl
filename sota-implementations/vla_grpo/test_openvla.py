# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the OpenVLA-OFT wrapper, using a tiny random-weight model.

The tiny model borrows the *structural* methods of the vendored
``OpenVLAForActionPrediction`` (input/label preparation, action masks,
multimodal assembly) so the wrapper is exercised against the real token
layout, while the heavy backbones are replaced by small random-weight
modules. No checkpoint download is involved. Run from this directory:

    pytest sota-implementations/vla_grpo/test_openvla.py

Requires ``transformers``, ``timm`` and ``Pillow`` (the vendored modeling
module imports them).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import NonTensorStack, TensorDict
from tensordict.nn import InteractionType
from torch import nn
from torchrl.data.vla import ACTION_TOKENS_KEY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils

_has_deps = all(
    importlib.util.find_spec(name) is not None
    for name in ("transformers", "timm", "PIL", "tokenizers")
)

if _has_deps:
    from openvla import OpenVLAOFTWrapper
    from openvla_oft.modeling_prismatic import OpenVLAForActionPrediction

CHUNK, ACT_DIM, N_BINS = 8, 7, 256
TRUE_VOCAB, PADDED_VOCAB, DIM = 32000, 32064, 16
LOG_PROBS_KEY = ("vla_action", "log_probs")


class _TinyVision(nn.Module):
    num_patches = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, kernel_size=8, stride=8)

    def forward(self, pixel_values):
        feats = self.proj(pixel_values.float())
        return feats.flatten(2).transpose(1, 2)

    def get_num_patches(self):
        return self.num_patches

    def get_num_images_in_input(self):
        return 1


class _TinyLM(nn.Module):
    # position-SENSITIVE on purpose: the action-token readout indexes
    # absolute positions, so a bag-of-tokens fake would mask indexing bugs
    # (e.g. reading pad positions in mixed-length batches)
    max_positions = 512

    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(DIM, DIM)
        self.head = nn.Linear(DIM, PADDED_VOCAB)
        self.positions = nn.Embedding(self.max_positions, DIM)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        hidden = inputs_embeds
        # masked positions contribute nothing, like a real attention mask
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        hidden = hidden * mask
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.positions(positions) * mask
        context = (hidden * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        logits = self.head(torch.tanh(self.mix(hidden + context)))
        return SimpleNamespace(logits=logits)


if _has_deps:

    class _TinyOFT(nn.Module):
        """Tiny random-weight stand-in sharing the vendored token layout."""

        # structural methods borrowed verbatim from the vendored class
        _prepare_input_for_action_prediction = (
            OpenVLAForActionPrediction._prepare_input_for_action_prediction
        )
        _prepare_labels_for_action_prediction = (
            OpenVLAForActionPrediction._prepare_labels_for_action_prediction
        )
        _process_action_masks = OpenVLAForActionPrediction._process_action_masks
        _process_vision_features = OpenVLAForActionPrediction._process_vision_features
        _build_multimodal_attention = (
            OpenVLAForActionPrediction._build_multimodal_attention
        )

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(PADDED_VOCAB, DIM)
            self.language_model = _TinyLM()
            self.vision_backbone = _TinyVision()
            self.projector = nn.Linear(DIM, DIM)
            self.vocab_size = TRUE_VOCAB
            bins = np.linspace(-1.0, 1.0, N_BINS)
            self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
            self.norm_stats = {
                "libero_spatial_no_noops": {
                    "action": {
                        "q01": [-0.5] * ACT_DIM,
                        "q99": [0.5] * ACT_DIM,
                        "mask": [True] * (ACT_DIM - 1) + [False],
                    }
                }
            }

        def get_input_embeddings(self):
            return self.embed


class _FakeProcessor:
    """Deterministic stand-in for PrismaticProcessor."""

    class _Tokenizer:
        pad_token_id = 0

    tokenizer = _Tokenizer()

    def __call__(self, prompt: str, image):
        data = prompt.encode("utf-8")
        length = 6 + len(data) % 9  # instruction-dependent prompt length
        ids = [1] + [3 + byte % 100 for byte in data[:length]]
        input_ids = torch.tensor([ids], dtype=torch.long)
        pixels = torch.as_tensor(
            np.asarray(image.resize((16, 16)), dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixels.unsqueeze(0),
        }


class _BatchTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, return_tensors="pt"):
        self.calls += 1
        data = prompt.encode("utf-8")
        ids = [1] + [3 + byte % 100 for byte in data[: 6 + len(data) % 9]]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class _BatchImageProcessor:
    input_sizes = [(3, 224, 224)]
    tvf_normalize_params = [{"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}]

    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors="pt"):
        self.calls += 1
        pixels = []
        for image in images:
            pixels.append(
                torch.from_numpy(np.asarray(image, dtype=np.uint8))
                .permute(2, 0, 1)
                .to(torch.float32)
                .div(255.0)
            )
        return {"pixel_values": torch.stack(pixels)}


class _BatchProcessor:
    """Batch-capable processor exposing the fast OpenVLA preprocessing hooks."""

    def __init__(self):
        self.tokenizer = _BatchTokenizer()
        self.image_processor = _BatchImageProcessor()

    def __call__(self, prompt, image):
        tokenized = self.tokenizer(prompt)
        image_features = self.image_processor([image])
        return {
            "input_ids": tokenized.input_ids,
            "pixel_values": image_features["pixel_values"],
        }


def _make_obs(batch=2, image_hw=32):
    instructions = [f"pick up object number {i}" * (i + 1) for i in range(batch)]
    return TensorDict(
        {
            "observation": {
                "image": torch.randint(
                    0, 256, (batch, 3, image_hw, image_hw), dtype=torch.uint8
                ),
            },
            "language_instruction": NonTensorStack(*instructions),
        },
        batch_size=[batch],
    )


@pytest.fixture
def policy():
    torch.manual_seed(0)
    return OpenVLAOFTWrapper(
        _TinyOFT(),
        _FakeProcessor(),
        temperature=1.6,
        default_interaction_type=InteractionType.RANDOM,
    )


def test_make_collector_casts_deviceless_env_to_cpu(monkeypatch):
    captured = {}

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    class _FakeEnv:
        batch_size = torch.Size([2])
        device = None

    cfg = SimpleNamespace(
        collector=SimpleNamespace(groups_per_iter=2, group_size=1),
        env=SimpleNamespace(max_outer_steps=3),
    )
    monkeypatch.setattr(utils, "Collector", _FakeCollector)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        collector = utils.make_collector(
            cfg, _FakeEnv(), object(), torch.device("cuda:0")
        )

    assert isinstance(collector, _FakeCollector)
    assert captured["kwargs"]["policy_device"] == torch.device("cuda:0")
    assert captured["kwargs"]["env_device"] == torch.device("cpu")


def test_make_collector_parallel_groups_use_logical_worker_count(monkeypatch):
    captured = {}

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    class _FakeEnv:
        batch_size = torch.Size([32])
        device = torch.device("cpu")

    cfg = SimpleNamespace(
        collector=SimpleNamespace(groups_per_iter=16, group_size=8),
        env=SimpleNamespace(max_outer_steps=3, parallel_group_repeats=True),
    )
    monkeypatch.setattr(utils, "Collector", _FakeCollector)

    with pytest.warns(UserWarning, match="parallel_group_repeats=true"):
        collector = utils.make_collector(
            cfg, _FakeEnv(), object(), torch.device("cuda:0")
        )

    assert isinstance(collector, _FakeCollector)
    assert captured["kwargs"]["trajs_per_batch"] == 16 * 8
    assert captured["kwargs"]["frames_per_batch"] == 32 * 3


def test_make_collector_parallel_group_wave_has_no_warning(monkeypatch):
    captured = {}

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    class _FakeEnv:
        batch_size = torch.Size([32])
        device = torch.device("cpu")

    cfg = SimpleNamespace(
        collector=SimpleNamespace(groups_per_iter=4, group_size=8),
        env=SimpleNamespace(max_outer_steps=3, parallel_group_repeats=True),
    )
    monkeypatch.setattr(utils, "Collector", _FakeCollector)

    collector = utils.make_collector(cfg, _FakeEnv(), object(), torch.device("cuda:0"))

    assert isinstance(collector, _FakeCollector)
    assert captured["kwargs"]["trajs_per_batch"] == 4 * 8
    assert captured["kwargs"]["frames_per_batch"] == 32 * 3


def test_make_collector_can_write_to_replay_buffer(monkeypatch):
    captured = {}

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    class _FakeEnv:
        batch_size = torch.Size([4])
        device = torch.device("cpu")

    cfg = SimpleNamespace(
        collector=SimpleNamespace(groups_per_iter=4, group_size=2),
        env=SimpleNamespace(max_outer_steps=3),
    )
    replay_buffer = object()

    def hook(_):
        return None

    monkeypatch.setattr(utils, "Collector", _FakeCollector)

    collector = utils.make_collector(
        cfg,
        _FakeEnv(),
        object(),
        torch.device("cpu"),
        replay_buffer=replay_buffer,
        post_collect_hook=hook,
    )

    assert isinstance(collector, _FakeCollector)
    assert captured["kwargs"]["replay_buffer"] is replay_buffer
    assert captured["kwargs"]["post_collect_hook"] is hook
    assert captured["kwargs"]["trajs_per_batch"] == 4 * 2
    assert captured["kwargs"]["frames_per_batch"] == 4


def test_make_replay_buffer_scales_capacity_with_overcollection():
    cfg = SimpleNamespace(
        collector=SimpleNamespace(
            groups_per_iter=2,
            group_size=3,
            max_collect_batches_per_iter=4,
        ),
        env=SimpleNamespace(max_outer_steps=5),
        advantage=SimpleNamespace(
            trajectory_return="sum",
            keep_return_bounds=None,
        ),
        loss=SimpleNamespace(mini_batch_size=2),
    )

    replay_buffer, _ = utils.make_replay_buffer(cfg, torch.device("cpu"))

    assert replay_buffer._storage.max_size == 2 * 3 * 5 * 4


def test_make_replay_buffer_scales_capacity_with_candidate_group_size():
    cfg = SimpleNamespace(
        collector=SimpleNamespace(
            groups_per_iter=2,
            group_size=3,
            candidate_group_size=6,
            max_collect_batches_per_iter=4,
        ),
        env=SimpleNamespace(max_outer_steps=5),
        advantage=SimpleNamespace(
            trajectory_return="sum",
            keep_return_bounds=None,
            candidate_selection="balanced",
            candidate_selection_min_size=4,
        ),
        loss=SimpleNamespace(mini_batch_size=2),
    )

    replay_buffer, advantage = utils.make_replay_buffer(cfg, torch.device("cpu"))

    assert replay_buffer._storage.max_size == 2 * 6 * 5 * 4
    assert advantage.grpo_size == 3
    assert advantage.candidate_group_size == 6
    assert advantage.candidate_selection_min_size == 4


def test_libero_worker_assignment_serial_and_parallel_groups():
    class _EnvCfg(dict):
        def __getattr__(self, name):
            return self[name]

    cfg = SimpleNamespace(
        collector=SimpleNamespace(group_size=8),
        env=_EnvCfg(
            {
                "task_ids": [10, 11, 12],
                "parallel_group_repeats": False,
            }
        ),
    )

    assert utils._libero_worker_assignment(cfg, 4, group_repeats=8) == (
        11,
        8,
        4 * utils.GROUP_ID_OFFSET,
    )

    cfg.env["parallel_group_repeats"] = True

    assert utils._libero_worker_assignment(cfg, 0, group_repeats=8) == (10, 1, 0)
    assert utils._libero_worker_assignment(cfg, 7, group_repeats=8) == (10, 1, 0)
    assert utils._libero_worker_assignment(cfg, 8, group_repeats=8) == (
        11,
        1,
        utils.GROUP_ID_OFFSET,
    )
    cfg.collector.candidate_group_size = 16
    assert utils._libero_worker_assignment(cfg, 7, group_repeats=8) == (10, 2, 0)


def test_make_libero_worker_parallel_groups_by_init_state(monkeypatch):
    class _Cfg(dict):
        def __getattr__(self, name):
            return self[name]

    captured = {}

    def _fake_libero_env(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    cfg = SimpleNamespace(
        env=_Cfg(
            task_suite="libero_spatial",
            task_ids=[0, 1],
            camera_height=64,
            camera_width=64,
            render_gpu_ids=[2, 3],
            eval_render_gpu_ids=None,
            max_env_steps=512,
            train_init_state_mode="cycle",
            parallel_group_repeats=True,
        ),
        collector=SimpleNamespace(group_size=8),
        policy=SimpleNamespace(use_wrist_image=False),
    )
    monkeypatch.setattr(utils, "LiberoEnv", _fake_libero_env)

    utils._make_libero_worker(cfg, 7, group_repeats=8)

    assert captured["args"] == ("libero_spatial",)
    assert captured["kwargs"]["task_id"] == 0
    assert captured["kwargs"]["group_repeats"] == 1
    assert captured["kwargs"]["group_id_offset"] == 0
    assert captured["kwargs"]["group_id_mode"] == "init_state"
    assert captured["kwargs"]["env_kwargs"]["render_gpu_device_id"] == 3


@pytest.mark.skipif(not _has_deps, reason="transformers/timm/PIL not found")
class TestOpenVLAOFTWrapper:
    def test_forward_shapes(self, policy):
        out = policy(_make_obs())
        assert out[ACTION_TOKENS_KEY].shape == (2, CHUNK, ACT_DIM)
        assert out[ACTION_TOKENS_KEY].min() >= 0
        assert out[ACTION_TOKENS_KEY].max() < N_BINS
        assert out[LOG_PROBS_KEY].shape == (2,)

    def test_temperature_contract_ratio_one(self, policy):
        # the go/no-go contract: with identical weights, the log-probs written
        # at rollout time and the loss-time recompute agree exactly, so the
        # PPO importance ratio is 1 even at T != 1
        obs = _make_obs()
        with torch.no_grad():
            out = policy(obs.clone())
            recomputed = policy.get_dist(obs.clone()).log_prob(out[ACTION_TOKENS_KEY])
        torch.testing.assert_close(recomputed, out[LOG_PROBS_KEY])

    def test_temperature_scales_logits(self):
        torch.manual_seed(0)
        model, processor = _TinyOFT(), _FakeProcessor()
        hot = OpenVLAOFTWrapper(model, processor, temperature=2.0)
        cold = OpenVLAOFTWrapper(model, processor, temperature=1.0)
        obs = _make_obs()
        with torch.no_grad():
            hot_logits = hot._action_logits(obs.clone())
            cold_logits = cold._action_logits(obs.clone())
        torch.testing.assert_close(hot_logits * 2.0, cold_logits)
        with pytest.raises(ValueError, match="temperature"):
            OpenVLAOFTWrapper(model, processor, temperature=0.0)

    def test_top_k_masks_logits(self):
        torch.manual_seed(0)
        policy = OpenVLAOFTWrapper(_TinyOFT(), _FakeProcessor(), top_k=3)
        logits = policy._action_logits(_make_obs())
        assert logits.isfinite().sum(-1).eq(3).all()
        with pytest.raises(ValueError, match="top_k"):
            OpenVLAOFTWrapper(_TinyOFT(), _FakeProcessor(), top_k=0)

    def test_greedy_deterministic(self):
        torch.manual_seed(0)
        policy = OpenVLAOFTWrapper(
            _TinyOFT(),
            _FakeProcessor(),
            default_interaction_type=InteractionType.DETERMINISTIC,
        )
        obs = _make_obs()
        with torch.no_grad():
            first = policy(obs.clone())[ACTION_TOKENS_KEY]
            second = policy(obs.clone())[ACTION_TOKENS_KEY]
            logits = policy._action_logits(obs.clone())
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, logits.argmax(-1))

    def test_mixed_prompt_lengths_consistent(self, policy):
        # heterogeneous batches pad shorter prompts: the action-token logits
        # of each row must be identical whether the row is forwarded alone or
        # inside a mixed-length batch (regression for the padding sort: pads
        # must be moved after the appended action placeholders, else the
        # readout hits pad positions for every row shorter than the batch max)
        obs = _make_obs(batch=3)  # instruction lengths differ per row
        with torch.no_grad():
            batched = policy._action_logits(obs.clone())
            for row in range(3):
                single = policy._action_logits(obs[row : row + 1].clone())
                torch.testing.assert_close(
                    batched[row : row + 1],
                    single,
                    msg=f"row {row} differs between batched and single forward",
                )

    def test_token_log_probs_mode(self):
        torch.manual_seed(0)
        policy = OpenVLAOFTWrapper(_TinyOFT(), _FakeProcessor(), log_probs_mode="token")
        out = policy(_make_obs())
        assert out[LOG_PROBS_KEY].shape == (2, CHUNK, ACT_DIM)

    def test_preprocess_batches_images_and_caches_prompts(self):
        torch.manual_seed(0)
        processor = _BatchProcessor()
        policy = OpenVLAOFTWrapper(_TinyOFT(), processor)
        images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        instructions = [
            "pick up the bowl",
            "pick up the bowl",
            "open drawer",
            "open drawer",
        ]

        input_ids, attention_mask, pixel_values = policy._preprocess(
            images, None, instructions
        )

        assert input_ids.shape[0] == len(instructions)
        assert attention_mask.shape == input_ids.shape
        assert pixel_values.shape == (4, 3, 224, 224)
        assert processor.tokenizer.calls == 2
        assert processor.image_processor.calls == 0

        policy._preprocess(images, None, instructions)

        assert processor.tokenizer.calls == 2

    def test_unbatched_input(self, policy):
        obs = TensorDict(
            {
                "observation": {
                    "image": torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
                },
                "language_instruction": "pick up the bowl",
            },
            batch_size=[],
        )
        out = policy(obs)
        assert out[ACTION_TOKENS_KEY].shape == (CHUNK, ACT_DIM)
        assert out[LOG_PROBS_KEY].shape == ()

    def test_action_tokenizer_decode(self, policy):
        tokenizer = policy.action_tokenizer
        assert tokenizer.vocab_size == N_BINS
        out = policy(_make_obs())
        actions = tokenizer.decode(out[ACTION_TOKENS_KEY])
        assert actions.shape == (2, CHUNK, ACT_DIM)
        # masked dims land inside the q01/q99 range, the gripper dim in [-1, 1]
        assert (actions[..., :-1] >= -0.5 - 1e-5).all()
        assert (actions[..., :-1] <= 0.5 + 1e-5).all()
        assert (actions[..., -1].abs() <= 1.0 + 1e-5).all()
        # decode -> encode is the identity on emitted tokens
        torch.testing.assert_close(tokenizer.encode(actions), out[ACTION_TOKENS_KEY])

    @pytest.mark.parametrize("ratio_level", ["sequence", "token"])
    def test_clip_ppo_loss_integration(self, ratio_level):
        from torchrl.objectives import ClipPPOLoss

        torch.manual_seed(0)
        policy = OpenVLAOFTWrapper(
            _TinyOFT(),
            _FakeProcessor(),
            temperature=1.6,
            default_interaction_type=InteractionType.RANDOM,
            log_probs_mode=ratio_level,
        )
        obs = _make_obs()
        with torch.no_grad():
            rollout = policy(obs.clone())
        data = obs.clone()
        data[ACTION_TOKENS_KEY] = rollout[ACTION_TOKENS_KEY]
        data[LOG_PROBS_KEY] = rollout[LOG_PROBS_KEY].detach()
        advantage = torch.tensor([[1.0], [-0.5]])
        if ratio_level == "token":
            # one ratio per token: the decision's advantage is broadcast over
            # the token dims (as the training script does)
            data["advantage"] = advantage.view(-1, 1, 1, 1).expand(
                *rollout[ACTION_TOKENS_KEY].shape, 1
            )
        else:
            data["advantage"] = advantage
        loss = ClipPPOLoss(
            policy,
            critic_network=None,
            entropy_bonus=False,
            clip_epsilon=(0.2, 0.28),
        )
        loss.set_keys(
            action=ACTION_TOKENS_KEY,
            sample_log_prob=LOG_PROBS_KEY,
            advantage="advantage",
        )
        out = loss(data)
        # identical weights: ratio == 1 everywhere, so the (negative) gain is
        # exactly the negated mean advantage and nothing is clipped
        torch.testing.assert_close(out["loss_objective"], -advantage.mean())
        assert out["clip_fraction"].item() == 0.0


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
