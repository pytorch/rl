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
from torchrl.data.vla.preprocessing import _fractional_center_crop, _lanczos3_resize

_has_pil = importlib.util.find_spec("PIL") is not None
_has_tensorflow = importlib.util.find_spec("tensorflow") is not None
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
    @pytest.mark.skipif(not _has_torchvision, reason="torchvision not found")
    def test_default_backend(self):
        proc = OpenVLAImagePreprocessor()

        assert proc.backend == "torch_reference"

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
        fast_again = tv(images)
        assert fast.shape == fast_again.shape == ref.shape == torch.Size([2, 3, 32, 32])
        assert fast.dtype == fast_again.dtype == ref.dtype == torch.uint8
        assert (fast.float() - ref.float()).abs().mean() < 25.0

    @pytest.mark.skipif(not _has_torchvision, reason="torchvision not found")
    @pytest.mark.parametrize("channels", [1, 3])
    def test_torch_reference_backend(self, channels):
        proc = OpenVLAImagePreprocessor(
            size=32, backend="torch_reference", center_crop=True
        )
        images = torch.full((2, channels, 25, 39), 73, dtype=torch.uint8)

        out = proc(images)
        unbatched = proc(images[0])

        assert out.shape == torch.Size([2, 3, 32, 32])
        assert unbatched.shape == torch.Size([3, 32, 32])
        assert out.dtype == unbatched.dtype == torch.uint8
        torch.testing.assert_close(out[0], unbatched)
        torch.testing.assert_close(out[:, 0], out[:, 1])
        torch.testing.assert_close(out[:, 1], out[:, 2])

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not _has_torchvision, reason="torchvision not found")
    def test_torch_reference_cuda_input(self):
        proc = OpenVLAImagePreprocessor(
            size=16, backend="torch_reference", center_crop=True
        )
        images = torch.randint(0, 256, (2, 3, 17, 23), dtype=torch.uint8, device="cuda")

        out = proc(images)

        assert out.device == images.device
        assert out.shape == (2, 3, 16, 16)

    def test_torch_reference_interpolation_matches_tensorflow_golden(self):
        image = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5) * 7
        expected_resize = torch.tensor(
            [
                [6.495448, 15.063237, 24.094267, 32.662056],
                [53.416706, 61.9845, 71.015526, 79.58331],
                [100.33795, 108.90575, 117.93678, 126.50456],
            ]
        ).reshape(1, 1, 3, 4)
        torch.testing.assert_close(
            _lanczos3_resize(image, (3, 4)),
            expected_resize,
            rtol=0,
            atol=1e-4,
        )
        expected_upscale = torch.tensor(
            [
                [-1.605574, 3.354574, 10.117878, 15.132814, 21.896118, 26.856264],
                [21.150263, 26.110410, 32.873714, 37.888645, 44.651955, 49.612103],
                [52.269077, 57.229220, 63.992523, 69.007450, 75.770770, 80.730910],
                [83.387886, 88.348030, 95.111336, 100.126270, 106.889580, 111.849724],
                [
                    106.143720,
                    111.103874,
                    117.867180,
                    122.882100,
                    129.645420,
                    134.605560,
                ],
            ]
        ).reshape(1, 1, 5, 6)
        torch.testing.assert_close(
            _lanczos3_resize(image, (5, 6)),
            expected_upscale,
            rtol=0,
            atol=2e-4,
        )

        crop_input = torch.tensor(
            [
                [0, 17, 41, 83],
                [11, 59, 101, 131],
                [29, 73, 149, 191],
                [47, 97, 211, 255],
            ],
            dtype=torch.uint8,
        ).reshape(1, 1, 4, 4)
        expected_crop = torch.tensor(
            [
                [2, 20, 45, 83],
                [15, 60, 101, 130],
                [31, 74, 146, 186],
                [49, 98, 203, 247],
            ],
            dtype=torch.uint8,
        ).reshape(1, 1, 4, 4)
        torch.testing.assert_close(
            _fractional_center_crop(crop_input, (4, 4)), expected_crop
        )

    def test_lanczos3_resize_batch_matches_sequential_and_compiles(self):
        generator = torch.Generator().manual_seed(0)
        images = torch.rand(4, 3, 37, 53, generator=generator)
        expected = torch.cat(
            [_lanczos3_resize(image.unsqueeze(0), (41, 59)) for image in images]
        )

        actual = _lanczos3_resize(images, (41, 59))
        compiled = torch.compile(_lanczos3_resize, backend="eager", fullgraph=True)(
            images, (41, 59)
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(compiled, expected)

    def test_fractional_center_crop_does_not_mutate_and_compiles(self):
        generator = torch.Generator().manual_seed(1)
        images = torch.randint(
            0, 256, (4, 3, 17, 23), dtype=torch.uint8, generator=generator
        ).to(torch.float32)
        original = images.clone()

        expected = _fractional_center_crop(images, (19, 21))
        compiled = torch.compile(
            _fractional_center_crop, backend="eager", fullgraph=True
        )(images, (19, 21))

        torch.testing.assert_close(images, original)
        torch.testing.assert_close(compiled, expected)

    @pytest.mark.skipif(not _has_tensorflow, reason="TensorFlow not found")
    def test_torch_reference_interpolation_matches_tensorflow(self):
        tensorflow = importlib.import_module("tensorflow")
        generator = torch.Generator().manual_seed(0)
        images = torch.randint(
            0, 256, (2, 3, 37, 53), dtype=torch.uint8, generator=generator
        )
        expected_resize = tensorflow.image.resize(
            images.permute(0, 2, 3, 1).numpy(),
            (32, 32),
            method="lanczos3",
            antialias=True,
        )
        expected_resize = torch.from_numpy(expected_resize.numpy().copy()).permute(
            0, 3, 1, 2
        )
        resize = _lanczos3_resize(images.float(), (32, 32))
        torch.testing.assert_close(resize, expected_resize, rtol=0, atol=2e-4)

        quantized = expected_resize.round().clamp(0, 255).to(torch.uint8)
        crop_size = tensorflow.sqrt(tensorflow.constant(0.9))
        offset = (1 - crop_size) / 2
        boxes = tensorflow.stack(
            [[offset, offset, offset + crop_size, offset + crop_size]] * 2
        )
        expected_crop = tensorflow.image.crop_and_resize(
            tensorflow.image.convert_image_dtype(
                quantized.permute(0, 2, 3, 1).numpy(), tensorflow.float32
            ),
            boxes,
            tensorflow.range(2),
            (32, 32),
        )
        expected_crop = tensorflow.image.convert_image_dtype(
            tensorflow.clip_by_value(expected_crop, 0, 1),
            tensorflow.uint8,
            saturate=True,
        )
        expected_crop = torch.from_numpy(expected_crop.numpy().copy()).permute(
            0, 3, 1, 2
        )
        crop = _fractional_center_crop(quantized, (32, 32))
        error = (crop.to(torch.int16) - expected_crop).abs()
        assert error.max() <= 1
        assert error.to(torch.float32).mean() < 1e-3

    @pytest.mark.skipif(
        not (_has_tensorflow and _has_torchvision),
        reason="TensorFlow/torchvision not found",
    )
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("center_crop", [False, True])
    def test_torch_reference_is_closer_to_tensorflow(self, channels, center_crop):
        generator = torch.Generator().manual_seed(10 + channels)
        images = torch.randint(
            0, 256, (1, channels, 37, 53), dtype=torch.uint8, generator=generator
        )
        tensorflow = OpenVLAImagePreprocessor(
            size=32, backend="tensorflow", center_crop=center_crop
        )(images)
        torch_reference = OpenVLAImagePreprocessor(
            size=32, backend="torch_reference", center_crop=center_crop
        )(images)
        torchvision = OpenVLAImagePreprocessor(
            size=32, backend="torchvision", center_crop=center_crop
        )(images)

        reference_error = (torch_reference.float() - tensorflow.float()).abs().mean()
        torchvision_error = (torchvision.float() - tensorflow.float()).abs().mean()
        assert reference_error < torchvision_error

    @pytest.mark.skipif(not _has_tensorflow, reason="TensorFlow not found")
    def test_tensorflow_reference_backend(self):
        proc = OpenVLAImagePreprocessor(size=32, backend="tensorflow", center_crop=True)
        images = torch.arange(3 * 24 * 40, dtype=torch.int64).reshape(1, 3, 24, 40)
        images = images.remainder(256).to(torch.uint8)
        first = proc(images)
        second = proc(images)
        assert first.shape == torch.Size([1, 3, 32, 32])
        assert first.dtype == torch.uint8
        torch.testing.assert_close(first, second)

    @pytest.mark.skipif(not _has_pil, reason="Pillow not found")
    def test_normalization(self):
        proc = OpenVLAImagePreprocessor(
            size=16, backend="pil", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        out = proc(torch.zeros(3, 16, 16, dtype=torch.uint8))
        assert out.shape == torch.Size([3, 16, 16])
        assert out.dtype == torch.float32

    @pytest.mark.skipif(not _has_pil, reason="Pillow not found")
    def test_fused_backbone_normalization(self):
        proc = OpenVLAImagePreprocessor(
            size=16,
            backend="pil",
            mean=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            std=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        )
        out = proc(torch.zeros(3, 16, 16, dtype=torch.uint8))
        assert out.shape == torch.Size([6, 16, 16])
        torch.testing.assert_close(out[:3], torch.zeros_like(out[:3]))
        torch.testing.assert_close(out[3:], -torch.ones_like(out[3:]))


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

    def test_cpu_decode_matches_openvla_numpy_reference_exactly(self):
        norm_low = [-0.7454732114076613, -0.6616071462631226]
        norm_high = [0.9375, 0.8758928775787354]
        tok = VocabTailActionTokenizer(
            256,
            full_vocab_size=32064,
            norm_low=norm_low,
            norm_high=norm_high,
            norm_mask=torch.ones(2, dtype=torch.bool),
        )
        tokens = torch.tensor([[32063, 31936], [31808, 32000]])
        bins = np.linspace(-1.0, 1.0, 256)
        centers = (bins[:-1] + bins[1:]) / 2.0
        indices = np.clip(32064 - tokens.numpy() - 1, 0, len(centers) - 1)
        normalized = centers[indices]
        expected = 0.5 * (normalized + 1.0) * (
            np.asarray(norm_high) - np.asarray(norm_low) + 1e-8
        ) + np.asarray(norm_low)
        decoded = tok.decode(tokens)
        # the affine runs in NumPy float64 for reference parity, then the
        # result is cast back to the tokenizer's float32 working dtype
        assert decoded.dtype == torch.float32
        np.testing.assert_array_equal(decoded.numpy(), expected.astype(np.float32))

    def test_decode_norm_stats_output_dtype_and_device(self):
        # the NumPy float64 detour must not leak float64 (or a device change)
        # into the returned action chunk
        norm_low = torch.tensor([-0.5, 0.0])
        norm_high = torch.tensor([0.5, 2.0])
        tok = VocabTailActionTokenizer(256, norm_low=norm_low, norm_high=norm_high)
        tokens = tok.encode(torch.tensor([[0.1, 1.0], [-0.2, 0.5]]))
        decoded = tok.decode(tokens)
        assert decoded.dtype == torch.float32
        assert decoded.device == tokens.device

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_decode_norm_stats_cuda(self):
        norm_low = torch.tensor([-0.5, 0.0])
        norm_high = torch.tensor([0.5, 2.0])
        tok = VocabTailActionTokenizer(
            256, norm_low=norm_low, norm_high=norm_high
        ).cuda()
        actions = torch.tensor([[0.1, 1.0], [-0.2, 0.5]], device="cuda")
        tokens = tok.encode(actions)
        assert tokens.device.type == "cuda"
        decoded = tok.decode(tokens)
        assert decoded.device.type == "cuda"
        assert decoded.dtype == torch.float32
        ref = VocabTailActionTokenizer(256, norm_low=norm_low, norm_high=norm_high)
        torch.testing.assert_close(decoded.cpu(), ref.decode(tokens.cpu()))

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
        # Unmasked OpenVLA gripper dims are already decoded in the normalized
        # [-1, 1] convention; SimpleVLA-RL signs that value before inverting
        # for LIBERO.
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
            gripper_binarize_threshold=0.0,
            gripper_invert=True,
        )
        tokens = base.encode(torch.tensor([[-0.25], [0.25]]))
        decoded = tok.decode(tokens)
        torch.testing.assert_close(
            decoded, torch.tensor([[1.0], [-1.0]], dtype=decoded.dtype)
        )

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
