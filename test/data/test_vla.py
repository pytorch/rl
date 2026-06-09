# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ``torchrl.data.vla`` primitives: the canonical VLA schema, the
robot-dataset metadata container and the action tokenizers."""
from __future__ import annotations

import argparse
import json

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from torchrl.data.vla import (
    ACTION_KEY,
    INSTRUCTION_KEY,
    RobotDatasetMetadata,
    validate_vla_tensordict,
)


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
