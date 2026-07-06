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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import NonTensorStack, TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from torch import nn
from torchrl.data.vla import ACTION_CHUNK_KEY, ACTION_TOKENS_KEY
from torchrl.objectives import ClipPPOLoss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils

_has_deps = all(
    importlib.util.find_spec(name) is not None
    for name in ("transformers", "timm", "PIL", "tokenizers")
)

if _has_deps:
    from openvla import (
        GripperPostProcessTransform,
        OpenVLAOFTWrapper,
    )
    from openvla_oft.modeling_prismatic import OpenVLAForActionPrediction

CHUNK, ACT_DIM, N_BINS = 8, 7, 256
TRUE_VOCAB, PADDED_VOCAB, DIM = 32000, 32064, 16
LOG_PROBS_KEY = ("vla_action", "log_probs")
_REAL_CHECKPOINT = os.environ.get("TORCHRL_OPENVLA_TEST_CHECKPOINT")
_REAL_OBSERVATIONS = os.environ.get("TORCHRL_OPENVLA_TEST_OBSERVATIONS")


def _fake_token_policy():
    return TensorDictModule(
        lambda action_tokens: action_tokens,
        in_keys=[ACTION_TOKENS_KEY],
        out_keys=[ACTION_TOKENS_KEY],
    )


class TestAdvantageMetrics:
    def test_local_advantage_worker_metrics_and_reset(self, monkeypatch):
        monkeypatch.setenv("TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES", "1")
        advantage = utils.MCAdvantage(
            grpo_size=2,
            prompt_key="group_id",
            trajectory_return="sum",
        )

        class FakeCollector:
            def __init__(self):
                self.calls = []

            def map_fn(self, method_name):
                self.calls.append(method_name)
                if method_name.endswith("get_stats"):
                    stats = advantage.get_stats()
                    stats.update(
                        completed_groups=2,
                        written_groups=1,
                        dropped_groups=1,
                        completed_trajectories=4,
                        completed_decisions=12,
                        successful_trajectories=1,
                        trajectory_return_sum=1.0,
                        trajectory_return_max=1.0,
                    )
                    return [stats]
                return [None]

        collector = FakeCollector()
        metrics = utils.advantage_metrics(advantage, collector)
        assert metrics["buffer/complete_groups"] == 2
        assert metrics["buffer/kept_groups"] == 1
        assert metrics["buffer/skipped_groups"] == 1
        assert metrics["collector/completed_trajectories"] == 4
        assert metrics["collector/successful_trajectories"] == 1

        utils.reset_collection_state(advantage, collector)
        assert collector.calls[-3:] == [
            "replay_buffer._transform[0].clear_queues",
            "replay_buffer._transform[0].reset_stats",
            "reset",
        ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _has_deps, reason="OpenVLA dependencies are missing")
@pytest.mark.skipif(
    not (_REAL_CHECKPOINT and _REAL_OBSERVATIONS),
    reason="set the real OpenVLA checkpoint and observation fixture paths",
)
class TestRealCheckpoint:
    def test_real_checkpoint_microbatch_tokens_and_actions_match(self):
        """Exercise microbatch equivalence with a real checkpoint when configured."""
        cfg = SimpleNamespace(
            policy=SimpleNamespace(
                backend="openvla",
                mode="tokens",
                checkpoint=_REAL_CHECKPOINT,
                unnorm_key="libero_spatial_no_noops",
                dataset_statistics=os.environ.get(
                    "TORCHRL_OPENVLA_TEST_DATASET_STATISTICS",
                    "moojink/openvla-7b-oft-finetuned-libero-spatial",
                ),
                dtype="bfloat16",
                temperature=0.7,
                top_k=None,
                use_wrist_image=False,
                center_crop=True,
                image_backend="tensorflow",
                gripper_binarize=True,
                gripper_binarize_threshold=0.0,
                gripper_invert=True,
                lora_rank=32,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            loss=SimpleNamespace(ratio_level="token"),
        )
        fixture = torch.load(_REAL_OBSERVATIONS, map_location="cpu", weights_only=False)
        rows = TensorDict(
            {
                "observation": {"image": fixture["images"][:8]},
                "language_instruction": NonTensorStack(*fixture["instructions"][:8]),
            },
            batch_size=[8],
        )
        policy = utils.make_policy(cfg, torch.device("cuda:0"))
        policy.eval()
        logits = {}
        try:
            with torch.inference_mode():
                for microbatch in (1, 2, 4, 8):
                    policy.model_transform.micro_batch_size = microbatch
                    logits[microbatch] = policy._action_logits(rows.clone(False))
            reference = logits[1]
            generator = torch.Generator(device=reference.device).manual_seed(1234)
            uniform = torch.rand(
                reference.shape,
                generator=generator,
                device=reference.device,
            ).clamp_(1e-7, 1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(uniform))
            reference_tokens = (reference + gumbel).argmax(-1)
            reference_log_probs = reference.log_softmax(-1).gather(
                -1, reference_tokens.unsqueeze(-1)
            )
            reference_actions = policy.action_tokenizer.decode(reference_tokens.cpu())
            reference_actions = policy.gripper_postprocess.postprocess(
                reference_actions
            )
            for microbatch in (2, 4, 8):
                assert torch.equal(reference, logits[microbatch])
                tokens = (logits[microbatch] + gumbel).argmax(-1)
                assert torch.equal(reference_tokens, tokens)
                log_probs = (
                    logits[microbatch].log_softmax(-1).gather(-1, tokens.unsqueeze(-1))
                )
                assert torch.equal(reference_log_probs, log_probs)
                actions = policy.action_tokenizer.decode(tokens.cpu())
                actions = policy.gripper_postprocess.postprocess(actions)
                assert torch.equal(reference_actions, actions)
        finally:
            del policy
            torch.cuda.empty_cache()


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


def _complete_collector_cfg(**collector_overrides):
    collector = {
        "groups_per_iter": 4,
        "group_size": 2,
        "candidate_group_size": None,
        "num_collectors": 2,
        "envs_per_collector": 4,
        "server_max_batch_size": 1,
        "server_min_batch_size": 1,
        "server_timeout": 0.01,
        "server_collect_stats": True,
        "server_stats_window_size": 1024,
        "policy_micro_batch_size": None,
        "max_inflight_per_env": 1,
        "num_threads": 1,
        "env_sub_threads": 1,
        "storing_device": "cpu",
        "use_buffers": None,
    }
    collector.update(collector_overrides)
    return SimpleNamespace(**collector)


def _complete_toy_env_cfg(**env_overrides):
    env = {
        "backend": "toy",
        "action_dim": 2,
        "state_dim": 4,
        "image_shape": (3, 8, 8),
        "render_size": 16,
        "success_steps": 2,
        "success_tol": 0.25,
        "max_outer_steps": 3,
        "num_envs": 4,
        "eval_num_envs": 1,
        "seed": 0,
        "parallel_group_repeats": True,
        "train_init_state_mode": "cycle",
        "render_backend": None,
        "render_gpu_ids": [2, 3],
        "eval_render_gpu_ids": None,
        "render_gpu_device_zero_fallback": True,
        "env_kwargs": None,
    }
    env.update(env_overrides)
    return SimpleNamespace(**env)


def _complete_logger_cfg(**logger_overrides):
    logger = {
        "eval_episodes": 4,
        "eval_backend": "thread",
        "eval_busy_policy": "skip",
        "record_video_single_task": False,
    }
    logger.update(logger_overrides)
    return SimpleNamespace(**logger)


class TestCollectorFactory:
    def test_make_collector_uses_multicollector_policy_server(self, monkeypatch):
        captured = {}

        class _FakeMultiCollector:
            def __init__(self, *args, **kwargs):
                captured["multi_args"] = args
                captured["multi_kwargs"] = kwargs

        class _FakeServer:
            def __init__(self, **kwargs):
                captured["server_kwargs"] = kwargs
                self.transport = kwargs["transport"]

            def start(self):
                captured["server_started"] = True
                return self

        class _FakeTransport:
            def __init__(self, **kwargs):
                captured["transport_kwargs"] = kwargs
                self.clients = []

            def client(self):
                client = object()
                self.clients.append(client)
                return client

        policy = _fake_token_policy()
        cfg = SimpleNamespace(
            collector=_complete_collector_cfg(policy_micro_batch_size=1),
            env=_complete_toy_env_cfg(),
        )
        replay_buffer = object()

        monkeypatch.setattr(utils, "MultiCollector", _FakeMultiCollector)
        monkeypatch.setattr(utils, "ProcessInferenceServer", _FakeServer)
        monkeypatch.setattr(utils, "MPTransport", _FakeTransport)

        collector, server, eval_policy = utils.make_collector(
            cfg,
            policy,
            torch.device("cpu"),
            tokenizer=utils.UniformActionTokenizer(16, low=-1.0, high=1.0),
            replay_buffer=replay_buffer,
        )

        assert isinstance(collector, _FakeMultiCollector)
        assert server is not None
        assert isinstance(eval_policy, utils.PolicyClientModule)
        assert captured["transport_kwargs"]["use_manager"]
        assert captured["server_started"]
        assert captured["server_kwargs"]["server_config"].max_batch_size == 1
        assert (
            captured["server_kwargs"]["policy_factory"].keywords[
                "policy_micro_batch_size"
            ]
            == 1
        )
        env_factories = captured["multi_args"][0]
        assert [
            factory.keywords["render_gpu_device_id"] for factory in env_factories
        ] == [
            2,
            3,
        ]
        assert [factory.keywords["worker_idx_offset"] for factory in env_factories] == [
            0,
            4,
        ]
        assert captured["multi_kwargs"]["policy"] is None
        assert len(captured["multi_kwargs"]["policy_factory"]) == 2
        assert captured["multi_kwargs"]["replay_buffer"] is replay_buffer
        assert captured["multi_kwargs"]["trajs_per_batch"] == 1
        assert captured["multi_kwargs"]["traj_format"] == "cat"
        assert captured["multi_kwargs"]["storing_device"] == torch.device("cpu")

    def test_make_collector_rejects_cross_subcollector_parallel_groups_without_shared_rb(
        self,
    ):
        policy = _fake_token_policy()
        cfg = SimpleNamespace(
            collector=_complete_collector_cfg(
                num_collectors=4,
                envs_per_collector=20,
                group_size=8,
                groups_per_iter=10,
            ),
            env=_complete_toy_env_cfg(parallel_group_repeats=True),
        )

        with pytest.raises(ValueError, match="collector.envs_per_collector"):
            utils.make_collector(
                cfg,
                policy,
                torch.device("cpu"),
                tokenizer=utils.UniformActionTokenizer(16, low=-1.0, high=1.0),
                replay_buffer=SimpleNamespace(shared=False),
            )

    def test_make_collector_allows_cross_subcollector_groups_with_shared_rb(
        self, monkeypatch
    ):
        captured = {}

        class _FakeMultiCollector:
            def __init__(self, *args, **kwargs):
                captured["multi_args"] = args
                captured["multi_kwargs"] = kwargs

        class _FakeServer:
            def __init__(self, **kwargs):
                self.transport = kwargs["transport"]

            def start(self):
                return self

        class _FakeTransport:
            def __init__(self, **kwargs):
                pass

            def client(self):
                return object()

        policy = _fake_token_policy()
        cfg = SimpleNamespace(
            collector=_complete_collector_cfg(
                num_collectors=4,
                envs_per_collector=20,
                group_size=8,
                groups_per_iter=10,
            ),
            env=_complete_toy_env_cfg(parallel_group_repeats=True),
        )
        replay_buffer = SimpleNamespace(shared=True)

        monkeypatch.setattr(utils, "MultiCollector", _FakeMultiCollector)
        monkeypatch.setattr(utils, "ProcessInferenceServer", _FakeServer)
        monkeypatch.setattr(utils, "MPTransport", _FakeTransport)

        collector, _, _ = utils.make_collector(
            cfg,
            policy,
            torch.device("cpu"),
            tokenizer=utils.UniformActionTokenizer(16, low=-1.0, high=1.0),
            replay_buffer=replay_buffer,
        )

        assert isinstance(collector, _FakeMultiCollector)
        assert len(captured["multi_args"][0]) == 4


class TestEvaluatorFactory:
    def test_make_evaluator_process_backend_uses_factories(self, monkeypatch):
        captured = {}

        class _FakeEvaluator:
            def __init__(self, env, policy=None, policy_factory=None, **kwargs):
                captured["env"] = env
                captured["policy"] = policy
                captured["policy_factory"] = policy_factory
                captured["kwargs"] = kwargs

        monkeypatch.setattr(utils, "Evaluator", _FakeEvaluator)

        policy = _fake_token_policy()
        cfg = SimpleNamespace(
            env=_complete_toy_env_cfg(),
            logger=_complete_logger_cfg(eval_backend="process"),
        )
        evaluator = utils.make_evaluator(
            cfg,
            utils.UniformActionTokenizer(16, low=-1.0, high=1.0),
            policy,
            logger=object(),
            device=torch.device("cpu"),
        )

        assert isinstance(evaluator, _FakeEvaluator)
        assert callable(captured["env"])
        assert captured["policy"] is None
        assert captured["policy_factory"]() is policy
        assert not captured["kwargs"]["dump_video"]
        assert captured["kwargs"]["backend"] == "process"
        assert captured["kwargs"]["max_steps"] == 0


class TestReplayBufferFactory:
    def test_make_replay_buffer_scales_capacity_with_overcollection(self):
        cfg = SimpleNamespace(
            collector=SimpleNamespace(
                groups_per_iter=2,
                group_size=3,
                candidate_group_size=None,
            ),
            env=SimpleNamespace(max_outer_steps=5),
            advantage=SimpleNamespace(
                trajectory_return="sum",
                keep_return_bounds=None,
                candidate_selection="balanced",
                candidate_selection_min_size=None,
                candidate_selection_max_combinations=100000,
            ),
            loss=SimpleNamespace(mini_batch_size=2),
            buffer=SimpleNamespace(
                shared_init=True,
                capacity_group_waves=4,
                consume_after_n_samples=1,
            ),
        )

        replay_buffer, _ = utils.make_replay_buffer(cfg, torch.device("cpu"))

        assert replay_buffer._storage.max_size == 2 * 3 * 5 * 4
        assert replay_buffer._storage.shared_init
        assert replay_buffer.shared

    def test_make_replay_buffer_scales_capacity_with_candidate_group_size(self):
        cfg = SimpleNamespace(
            collector=SimpleNamespace(
                groups_per_iter=2,
                group_size=3,
                candidate_group_size=6,
            ),
            env=SimpleNamespace(max_outer_steps=5),
            advantage=SimpleNamespace(
                trajectory_return="sum",
                keep_return_bounds=None,
                candidate_selection="balanced",
                candidate_selection_min_size=4,
                candidate_selection_max_combinations=100000,
            ),
            loss=SimpleNamespace(mini_batch_size=2),
            buffer=SimpleNamespace(
                shared_init=False,
                capacity_group_waves=4,
                consume_after_n_samples=1,
            ),
        )

        replay_buffer, advantage = utils.make_replay_buffer(cfg, torch.device("cpu"))

        assert replay_buffer._storage.max_size == 2 * 6 * 5 * 4
        assert advantage.grpo_size == 3
        assert advantage.candidate_group_size == 6
        assert advantage.candidate_selection_min_size == 4


class TestTokenizerAndTransforms:


    def test_chunk_transform_without_tokenizer_consumes_vla_chunk(self):
        cfg = SimpleNamespace(env=SimpleNamespace(max_outer_steps=1))

        transform = utils._chunk_transform(cfg, None)

        assert transform[0].out_keys_inv == [ACTION_CHUNK_KEY]

    def test_chunk_transform_openvla_tokens_decodes_then_postprocesses(self):
        cfg = SimpleNamespace(
            env=SimpleNamespace(max_outer_steps=1),
            policy=SimpleNamespace(
                backend="openvla",
                mode="tokens",
                gripper_binarize=True,
                gripper_binarize_threshold=0.0,
                gripper_invert=True,
            ),
        )

        transform = utils._chunk_transform(
            cfg,
            utils.UniformActionTokenizer(16, low=-1.0, high=1.0),
        )

        assert transform[0].__class__.__name__ == "MultiAction"
        assert transform[1].__class__.__name__ == "GripperPostProcessTransform"
        assert transform[2].__class__.__name__ == "ActionTokenizerTransform"
        assert transform[1].in_keys_inv == ["action"]
        assert transform[1].out_keys_inv == ["action"]


class TestLiberoWorkers:
    def test_libero_worker_assignment_serial_and_parallel_groups(self):
        class _EnvCfg(dict):
            def __getattr__(self, name):
                return self[name]

        cfg = SimpleNamespace(
            collector=SimpleNamespace(group_size=8, candidate_group_size=None),
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

    def test_make_libero_worker_parallel_groups_by_init_state(self, monkeypatch):
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
                render_backend="egl",
                render_gpu_ids=[2, 3],
                eval_render_gpu_ids=None,
                render_gpu_device_zero_fallback=True,
                env_kwargs=None,
                max_env_steps=512,
                train_init_state_mode="cycle",
                train_init_state_id=3,
                parallel_group_repeats=True,
            ),
            collector=SimpleNamespace(group_size=8, candidate_group_size=None),
            policy=SimpleNamespace(use_wrist_image=False),
        )
        monkeypatch.setattr(utils, "LiberoEnv", _fake_libero_env)

        utils._make_libero_worker(cfg, 7, group_repeats=8)

        assert captured["args"] == ("libero_spatial",)
        assert captured["kwargs"]["task_id"] == 0
        assert captured["kwargs"]["group_repeats"] == 1
        assert captured["kwargs"]["group_id_offset"] == 0
        assert captured["kwargs"]["group_id_mode"] == "init_state"
        assert captured["kwargs"]["init_state_id"] == 3
        assert captured["kwargs"]["env_kwargs"]["render_gpu_device_id"] == 3


@pytest.mark.skipif(not _has_deps, reason="transformers/timm/PIL not found")
class TestOpenVLAOFTWrapper:
    def test_forward_shapes(self, policy):
        out = policy(_make_obs())
        assert out[ACTION_TOKENS_KEY].shape == (2, CHUNK, ACT_DIM)
        assert out[ACTION_TOKENS_KEY].min() >= 0
        assert out[ACTION_TOKENS_KEY].max() < N_BINS
        assert out[LOG_PROBS_KEY].shape == (2,)

    def test_forward_nested_batch_shapes(self, policy):
        out = policy(_make_obs(batch=8).reshape(1, 8))
        assert out[ACTION_TOKENS_KEY].shape == (1, 8, CHUNK, ACT_DIM)
        assert out[LOG_PROBS_KEY].shape == (1, 8)

    def test_policy_stack_matches_action_logits(self, policy):
        obs = _make_obs()

        with torch.no_grad():
            stacked = policy.policy_stack(obs.clone())
            logits = policy._action_logits(obs.clone())

        torch.testing.assert_close(
            stacked.get(policy.tensor_keys.action_logits),
            logits,
        )
        assert stacked.get(policy.input_transform.input_ids_key).shape[0] == 2
        assert stacked.get(policy.input_transform.pixel_values_key).shape[0] == 2

    def test_gripper_postprocess_matches_simplevla_order(self):
        transform = GripperPostProcessTransform(
            action_key="action",
            rescale=True,
            binarize=True,
            threshold=0.0,
            invert=True,
        )
        actions = torch.zeros(2, CHUNK, ACT_DIM)
        actions[0, :, -1] = 0.25
        actions[1, :, -1] = 0.75

        out = transform(TensorDict({"action": actions}, batch_size=[2]))["action"]

        torch.testing.assert_close(out[0, :, -1], torch.ones(CHUNK))
        torch.testing.assert_close(out[1, :, -1], -torch.ones(CHUNK))

    def test_decode_stack_applies_tokenizer_then_gripper_postprocess(self):
        policy = OpenVLAOFTWrapper(
            _TinyOFT(),
            _FakeProcessor(),
            output_mode="both",
            gripper_binarize=True,
            gripper_invert=True,
        )
        obs = _make_obs()

        with torch.no_grad():
            out = policy(obs)

        decoded = policy.action_tokenizer.decode(out[ACTION_TOKENS_KEY])
        expected = policy.gripper_postprocess.postprocess(decoded)
        torch.testing.assert_close(out[ACTION_CHUNK_KEY], expected)

        decoded_td = policy.decode_stack(
            TensorDict(
                {ACTION_TOKENS_KEY: out[ACTION_TOKENS_KEY]},
                batch_size=out.batch_size,
            )
        )
        torch.testing.assert_close(decoded_td[ACTION_CHUNK_KEY], expected)

    def test_gripper_postprocess_skips_observation_forward_path(self):
        transform = GripperPostProcessTransform(action_key="action")
        observation = TensorDict({"observation": torch.zeros(3)}, batch_size=[])
        out = transform(observation)
        assert "action" not in out
        torch.testing.assert_close(out["observation"], torch.zeros(3))

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

    def test_model_microbatch_matches_sequential(self):
        policy = OpenVLAOFTWrapper(_TinyOFT(), _FakeProcessor(), micro_batch_size=1)
        obs = _make_obs(batch=3)
        with torch.no_grad():
            microbatched = policy._action_logits(obs.clone())
            sequential = torch.cat(
                [policy._action_logits(obs[row : row + 1].clone()) for row in range(3)],
                dim=0,
            )
        torch.testing.assert_close(microbatched, sequential)
        assert policy.model_transform.micro_batch_size == 1
        with pytest.raises(ValueError, match="micro_batch_size"):
            OpenVLAOFTWrapper(_TinyOFT(), _FakeProcessor(), micro_batch_size=0)

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
        torch.manual_seed(0)
        policy = OpenVLAOFTWrapper(
            _TinyOFT(),
            _FakeProcessor(),
            temperature=1.6,
            default_interaction_type=InteractionType.RANDOM,
            log_probs_mode=ratio_level,
        )
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
        ess = []
        for batch_size in (2, 1):
            obs = _make_obs(batch=batch_size)
            with torch.no_grad():
                rollout = policy(obs.clone())
            data = obs.clone()
            data[ACTION_TOKENS_KEY] = rollout[ACTION_TOKENS_KEY]
            data[LOG_PROBS_KEY] = rollout[LOG_PROBS_KEY].detach()
            advantage = torch.linspace(1.0, -0.5, batch_size).unsqueeze(-1)
            if ratio_level == "token":
                # one ratio per token: the decision's advantage is broadcast over
                # the token dims (as the training script does)
                data["advantage"] = advantage.view(-1, 1, 1, 1).expand(
                    *rollout[ACTION_TOKENS_KEY].shape, 1
                )
                expected_ess_shape = (CHUNK, ACT_DIM)
            else:
                data["advantage"] = advantage
                expected_ess_shape = ()

            out = loss(data)
            # identical weights: ratio == 1 everywhere, so the (negative) gain is
            # exactly the negated mean advantage and nothing is clipped
            torch.testing.assert_close(out["loss_objective"], -advantage.mean())
            assert out["clip_fraction"].item() == 0.0
            assert out["ESS"].shape == expected_ess_shape
            torch.testing.assert_close(out["ESS"], torch.ones_like(out["ESS"]))
            ess.append(out["ESS"].detach().mean())

        assert torch.stack(ess).shape == (2,)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
