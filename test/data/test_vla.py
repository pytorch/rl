# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ``torchrl.data.vla`` primitives: the canonical VLA schema, the
robot-dataset metadata container and the action tokenizers."""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, TensorDict

from torchrl.data.vla import (
    ACTION_KEY,
    ActionTokenizerBase,
    INSTRUCTION_KEY,
    OpenVLAImagePreprocessor,
    RobotDatasetMetadata,
    UniformActionTokenizer,
    validate_vla_tensordict,
    VLAAction,
    VLAImages,
    VLAObservation,
    VocabTailActionTokenizer,
)

_has_pil = importlib.util.find_spec("PIL") is not None
_has_torchvision = importlib.util.find_spec("torchvision") is not None


def _make_vla_td(batch=2, action_dim=7, *, with_instruction=True, finite=True):
    obs = {"image": torch.zeros(batch, 3, 8, 8, dtype=torch.uint8)}
    action = torch.zeros(batch, action_dim)
    if not finite:
        action[0, 0] = float("nan")
    data = {"observation": obs, "action": action}
    # The canonical schema keeps the (per-trajectory) instruction at the root,
    # mirroring OpenXExperienceReplay.
    if with_instruction:
        data["language_instruction"] = NonTensorData("pick up the red block")
    return TensorDict(data, batch_size=[batch])


class TestRobotDatasetMetadata:
    def test_basic(self):
        meta = RobotDatasetMetadata(
            "bridge",
            action_dim=7,
            action_space="eef_delta",
            gripper_mode="binary",
            control_frequency_hz=5.0,
        )
        assert meta.dataset_id == "bridge"
        assert meta.action_dim == 7
        assert meta.action_key == ACTION_KEY

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"action_space": "not-a-space"},
            {"gripper_mode": "weird"},
            {"action_dim": 0},
            {"action_dim": -3},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            RobotDatasetMetadata("bridge", **kwargs)

    def test_make_action_spec_unbounded(self):
        meta = RobotDatasetMetadata("bridge", action_dim=7)
        spec = meta.make_action_spec()
        assert spec.shape == torch.Size([7])
        spec_b = meta.make_action_spec(shape_prefix=(4,))
        assert spec_b.shape == torch.Size([4, 7])

    def test_make_action_spec_bounded(self):
        meta = RobotDatasetMetadata(
            "bridge",
            action_dim=3,
            action_low=-torch.ones(3),
            action_high=torch.ones(3),
        )
        spec = meta.make_action_spec()
        assert spec.shape == torch.Size([3])
        # values drawn from a bounded spec respect the bounds
        sample = spec.rand()
        assert (sample >= -1).all() and (sample <= 1).all()

    def test_make_action_spec_bounded_with_prefix(self):
        meta = RobotDatasetMetadata(
            "bridge", action_dim=3, action_low=-torch.ones(3), action_high=torch.ones(3)
        )
        spec = meta.make_action_spec(shape_prefix=(4,))
        assert spec.shape == torch.Size([4, 3])
        sample = spec.rand()
        assert (sample >= -1).all() and (sample <= 1).all()
        spec2 = meta.make_action_spec(shape_prefix=(2, 5))
        assert spec2.shape == torch.Size([2, 5, 3])

    def test_make_action_spec_requires_dim(self):
        with pytest.raises(ValueError, match="action_dim must be set"):
            RobotDatasetMetadata("bridge").make_action_spec()

    def test_action_dim_float_rejected(self):
        with pytest.raises(ValueError, match="positive int"):
            RobotDatasetMetadata("bridge", action_dim=7.0)

    def test_is_tensorclass(self):
        # RobotDatasetMetadata is a TensorClass: scalar batch, tensors move with it
        meta = RobotDatasetMetadata("d", action_dim=2, action_mean=torch.zeros(2))
        assert meta.batch_size == torch.Size([])
        assert meta.to("cpu").action_mean.device.type == "cpu"

    def test_from_json_ignores_unknown(self):
        meta = RobotDatasetMetadata("bridge", action_dim=4)
        data = json.loads(meta.to_json())
        data["some_future_field"] = 123
        assert (RobotDatasetMetadata.from_json(json.dumps(data)) == meta).all()

    def test_stats_coerced_to_tensor(self):
        meta = RobotDatasetMetadata(
            "bridge",
            action_dim=3,
            action_mean=[0.0, 1.0, 2.0],
            action_std=[1.0, 1.0, 1.0],
        )
        assert isinstance(meta.action_mean, torch.Tensor)
        assert meta.action_mean.dtype == torch.float32

    def test_json_roundtrip_preserves_fields(self):
        meta = RobotDatasetMetadata(
            "bridge",
            embodiment_id="widowx",
            action_dim=7,
            action_names=("x", "y", "z", "rx", "ry", "rz", "grip"),
            action_space="eef_delta",
            gripper_mode="binary",
            camera_keys=(
                ("observation", "image", "wrist"),
                ("observation", "image", "front"),
            ),
            action_mean=torch.arange(7, dtype=torch.float32),
            action_std=torch.ones(7),
        )
        meta2 = RobotDatasetMetadata.from_json(meta.to_json())
        assert (meta2 == meta).all()
        # nested camera keys survive as tuples (not lists)
        assert all(isinstance(k, tuple) for k in meta2.camera_keys)
        assert meta2.action_names == meta.action_names

    def test_json_roundtrip(self, tmp_path):
        meta = RobotDatasetMetadata(
            "bridge",
            action_dim=4,
            action_mean=torch.randn(4),
            action_std=torch.rand(4) + 0.1,
        )
        assert (RobotDatasetMetadata.from_json(meta.to_json()) == meta).all()
        path = tmp_path / "meta.json"
        meta.to_json(str(path))
        assert (RobotDatasetMetadata.from_json(str(path)) == meta).all()

    def test_eq_handles_tensors(self):
        a = RobotDatasetMetadata("d", action_dim=2, action_mean=torch.zeros(2))
        b = RobotDatasetMetadata("d", action_dim=2, action_mean=torch.zeros(2))
        c = RobotDatasetMetadata("d", action_dim=2, action_mean=torch.ones(2))
        assert (a == b).all()
        assert not (a == c).all()


class TestVLASchema:
    def test_valid(self):
        assert validate_vla_tensordict(_make_vla_td()) == []

    def test_missing_instruction(self):
        td = _make_vla_td(with_instruction=False)
        issues = validate_vla_tensordict(td, raise_on_error=False)
        assert any("language instruction" in i for i in issues)
        with pytest.raises(ValueError, match="VLA schema"):
            validate_vla_tensordict(td)

    def test_missing_instruction_allowed(self):
        td = _make_vla_td(with_instruction=False)
        assert validate_vla_tensordict(td, require_instruction=False) == []

    def test_non_finite_action(self):
        td = _make_vla_td(finite=False)
        issues = validate_vla_tensordict(td, raise_on_error=False)
        assert any("non-finite" in i for i in issues)

    def test_empty_instruction(self):
        td = _make_vla_td()
        td.set(INSTRUCTION_KEY, NonTensorData("   "))
        issues = validate_vla_tensordict(td, raise_on_error=False)
        assert any("empty language" in i for i in issues)

    def test_no_perception(self):
        td = TensorDict(
            {
                "observation": {},
                "language_instruction": NonTensorData("do it"),
                "action": torch.zeros(2, 7),
            },
            batch_size=[2],
        )
        issues = validate_vla_tensordict(td, raise_on_error=False)
        assert any("no perception" in i for i in issues)

    def test_descend_into_leaf_image(self):
        # image is a single tensor; probing a named camera under it (a NestedKey
        # that descends into a leaf) must report missing perception, not crash.
        td = _make_vla_td()
        issues = validate_vla_tensordict(
            td, image_key=("observation", "image", "wrist"), raise_on_error=False
        )
        assert any("no perception" in i for i in issues)

    def test_descend_into_leaf_action(self):
        td = _make_vla_td()
        issues = validate_vla_tensordict(
            td, action_key=("action", "sub"), raise_on_error=False
        )
        assert any("missing action" in i for i in issues)

    def test_nested_custom_keys(self):
        td = TensorDict(
            {
                "obs": {"cam": torch.zeros(2, 3, 4, 4, dtype=torch.uint8)},
                "instr": NonTensorData("go"),
                "act": torch.zeros(2, 6),
            },
            batch_size=[2],
        )
        assert (
            validate_vla_tensordict(
                td,
                instruction_key="instr",
                action_key="act",
                image_key=("obs", "cam"),
                state_key=("obs", "state"),
            )
            == []
        )


class TestVLAContainers:
    def test_action_container(self):
        action = VLAAction(
            chunk=torch.zeros(2, 4, 7),
            tokens=torch.zeros(2, 4, 7, dtype=torch.long),
            batch_size=[2],
        )
        assert action.chunk.shape == torch.Size([2, 4, 7])
        assert action.tokens.dtype == torch.long

    def test_observation_container(self):
        images = VLAImages(
            image=torch.zeros(2, 3, 16, 16, dtype=torch.uint8), batch_size=[2]
        )
        obs = VLAObservation(
            images=images,
            state=torch.zeros(2, 5),
            instruction=["pick", "place"],
            batch_size=[2],
        )
        assert obs.images.image.shape == torch.Size([2, 3, 16, 16])
        assert obs.state.shape == torch.Size([2, 5])


class TestOpenVLAImagePreprocessor:
    @pytest.mark.skipif(not _has_pil, reason="Pillow not found")
    def test_pil_backend_shapes(self):
        proc = OpenVLAImagePreprocessor(size=32, backend="pil", center_crop=True)
        images = torch.randint(0, 256, (2, 3, 24, 40), dtype=torch.uint8)
        out = proc(images)
        assert out.shape == torch.Size([2, 3, 32, 32])
        assert out.dtype == torch.uint8

    @pytest.mark.skipif(
        not (_has_pil and _has_torchvision), reason="Pillow/torchvision not found"
    )
    def test_torchvision_backend_matches_reference_shape(self):
        images = torch.randint(0, 256, (2, 3, 24, 40), dtype=torch.uint8)
        pil = OpenVLAImagePreprocessor(size=32, backend="pil", center_crop=False)
        tv = OpenVLAImagePreprocessor(size=32, backend="torchvision", center_crop=False)
        ref = pil(images)
        fast = tv(images)
        assert fast.shape == ref.shape == torch.Size([2, 3, 32, 32])
        assert fast.dtype == ref.dtype == torch.uint8
        assert (fast.float() - ref.float()).abs().mean() < 25.0

    @pytest.mark.skipif(not _has_pil, reason="Pillow not found")
    def test_normalization(self):
        proc = OpenVLAImagePreprocessor(
            size=16, backend="pil", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        out = proc(torch.zeros(3, 16, 16, dtype=torch.uint8))
        assert out.shape == torch.Size([3, 16, 16])
        assert out.dtype == torch.float32


class TestActionTokenizer:
    def test_encode_values(self):
        tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
        tokens = tok.encode(torch.tensor([-1.0, 0.0, 1.0]))
        assert tokens.tolist() == [0, 128, 255]
        assert tokens.dtype == torch.long

    def test_roundtrip_tolerance(self):
        tok = UniformActionTokenizer(64, low=-2.0, high=2.0)
        actions = torch.empty(1000).uniform_(-2.0, 2.0)
        recon = tok.decode(tok.encode(actions))
        bin_width = (2.0 - (-2.0)) / 64
        assert (recon - actions).abs().max() <= bin_width / 2 + 1e-5

    def test_is_base_subclass(self):
        assert issubclass(UniformActionTokenizer, ActionTokenizerBase)

    def test_vocab_and_action_dim(self):
        tok = UniformActionTokenizer(10, low=0.0, high=1.0)
        assert tok.vocab_size == 10
        assert tok.action_dim is None  # scalar bounds
        tok2 = UniformActionTokenizer(10, low=0.0, high=1.0, action_dim=4)
        assert tok2.action_dim == 4
        assert tok2.low.shape == torch.Size([4])

    def test_per_dim_bounds_chunk(self):
        low = torch.tensor([-1.0, 0.0, -2.0])
        high = torch.tensor([1.0, 2.0, 0.0])
        tok = UniformActionTokenizer(32, low=low, high=high)
        actions = torch.rand(2, 4, 5, 3)  # [B, T, chunk, A]
        actions = low + actions * (high - low)
        tokens = tok.encode(actions)
        assert tokens.shape == torch.Size([2, 4, 5, 3])
        recon = tok.decode(tokens)
        assert (recon - actions).abs().max() <= ((high - low) / (2 * 32)).max() + 1e-5

    def test_invalid_num_bins(self):
        with pytest.raises(ValueError, match="num_bins"):
            UniformActionTokenizer(0, low=-1.0, high=1.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="strictly greater"):
            UniformActionTokenizer(10, low=1.0, high=1.0)

    def test_from_metadata(self):
        meta = RobotDatasetMetadata(
            "bridge",
            action_dim=2,
            action_low=torch.tensor([-1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0]),
        )
        tok = UniformActionTokenizer.from_metadata(meta, num_bins=128)
        assert tok.vocab_size == 128
        assert tok.action_dim == 2

    def test_from_metadata_no_bounds_raises(self):
        meta = RobotDatasetMetadata("bridge", action_dim=2)
        with pytest.raises(ValueError, match="no action bounds"):
            UniformActionTokenizer.from_metadata(meta, num_bins=128)


class TestVocabTailActionTokenizer:
    def test_matches_openvla_reference(self):
        # oracle: the OpenVLA ActionTokenizer formulas (numpy), see
        # https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
        vocab, n_bins = 32000, 256
        tok = VocabTailActionTokenizer(n_bins, full_vocab_size=vocab)
        bins = tok.bins.numpy()
        centers = (bins[:-1] + bins[1:]) / 2.0
        actions = torch.linspace(-1.3, 1.3, 1001)
        ref_tokens = vocab - np.digitize(np.clip(actions.numpy(), -1.0, 1.0), bins)
        tokens = tok.encode(actions)
        assert (tokens.numpy() == ref_tokens).all()
        ref_decoded = centers[np.clip(vocab - ref_tokens - 1, 0, centers.shape[0] - 1)]
        torch.testing.assert_close(
            tok.decode(tokens), torch.as_tensor(ref_decoded, dtype=torch.float32)
        )

    def test_window_vs_full_convention(self):
        vocab, n_bins = 32000, 256
        window = VocabTailActionTokenizer(n_bins)
        full = VocabTailActionTokenizer(n_bins, full_vocab_size=vocab)
        actions = torch.linspace(-1.0, 1.0, 257)
        window_tokens = window.encode(actions)
        full_tokens = full.encode(actions)
        assert (full_tokens == window_tokens + vocab - n_bins).all()
        assert window_tokens.min() >= 0
        assert window_tokens.max() < n_bins
        torch.testing.assert_close(
            window.decode(window_tokens), full.decode(full_tokens)
        )

    def test_roundtrip_tolerance(self):
        tok = VocabTailActionTokenizer(256)
        actions = torch.empty(1000).uniform_(-1.0, 1.0)
        recon = tok.decode(tok.encode(actions))
        bin_width = 2.0 / 255
        assert (recon - actions).abs().max() <= bin_width / 2 + 1e-5

    def test_norm_stats_unnormalization(self):
        # masked dims roundtrip through the q01/q99 affine map; unmasked
        # (gripper) dims pass through in [-1, 1]
        norm_low = torch.tensor([-0.5, 0.0, -1.0])
        norm_high = torch.tensor([0.5, 2.0, -1.0 + 2.0])
        mask = torch.tensor([True, True, False])
        tok = VocabTailActionTokenizer(
            256, norm_low=norm_low, norm_high=norm_high, norm_mask=mask
        )
        actions = torch.tensor([[-0.25, 1.5, 1.0], [0.4, 0.1, -1.0]])
        recon = tok.decode(tok.encode(actions))
        scale = (norm_high - norm_low) / 255
        assert (recon[:, :2] - actions[:, :2]).abs().max() <= scale.max() / 2 + 1e-5
        # gripper dim decodes near the un-normalized [-1, 1] poles
        assert (recon[:, 2] - actions[:, 2]).abs().max() <= 2.0 / 255 + 1e-5

    def test_from_norm_stats(self):
        norm_stats = {
            "libero_spatial_no_noops": {
                "action": {
                    "q01": [-0.5] * 7,
                    "q99": [0.5] * 7,
                    "mask": [True] * 6 + [False],
                }
            }
        }
        tok = VocabTailActionTokenizer.from_norm_stats(
            norm_stats, "libero_spatial_no_noops", full_vocab_size=32000
        )
        assert tok.vocab_size == 32000
        assert tok.norm_mask.tolist() == [True] * 6 + [False]
        with pytest.raises(KeyError, match="unnorm_key"):
            VocabTailActionTokenizer.from_norm_stats(norm_stats, "bad_key")

    def test_gripper_binarize_threshold(self):
        # SimpleVLA-RL's LIBERO gripper post-processing normalizes a [0, 1]
        # gripper to [-1, 1] before taking the sign, which is equivalent to
        # thresholding the decoded gripper at 0.5.
        norm_low = torch.tensor([0.0])
        norm_high = torch.tensor([1.0])
        mask = torch.tensor([False])
        base = VocabTailActionTokenizer(
            256, norm_low=norm_low, norm_high=norm_high, norm_mask=mask
        )
        tok = VocabTailActionTokenizer(
            256,
            norm_low=norm_low,
            norm_high=norm_high,
            norm_mask=mask,
            gripper_binarize=True,
            gripper_binarize_threshold=0.5,
            gripper_invert=True,
        )
        tokens = base.encode(torch.tensor([[0.25], [0.75]]))
        torch.testing.assert_close(tok.decode(tokens), torch.tensor([[1.0], [-1.0]]))

    def test_validation(self):
        with pytest.raises(ValueError, match="num_bins"):
            VocabTailActionTokenizer(1)
        with pytest.raises(ValueError, match="full_vocab_size"):
            VocabTailActionTokenizer(256, full_vocab_size=128)
        with pytest.raises(ValueError, match="together"):
            VocabTailActionTokenizer(256, norm_low=torch.zeros(3))

    def test_chunk_shapes(self):
        tok = VocabTailActionTokenizer(256)
        actions = torch.empty(2, 4, 8, 7).uniform_(-1.0, 1.0)
        tokens = tok.encode(actions)
        assert tokens.shape == actions.shape
        assert tokens.dtype == torch.long
        assert tok.decode(tokens).shape == actions.shape

    def test_is_base_subclass(self):
        assert issubclass(VocabTailActionTokenizer, ActionTokenizerBase)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
