# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import doctest
import importlib.util
import os
import shutil
import time
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import (
    assert_allclose_td,
    MetaData,
    NonTensorData,
    NonTensorStack,
    TensorDict,
    TensorDictBase,
    UnbatchedTensor,
)
from torch.utils.data import DataLoader, Dataset, Subset

from torchrl._utils import logger as torchrl_logger
from torchrl.data import (
    LazyTensorStorage,
    ListStorage,
    ReplayBuffer,
    ReplayBufferEnsemble,
    SamplerWithoutReplacement,
    tensordict_collate,
    TensorStorage,
    VideoClipRef,
)
from torchrl.data.datasets import (
    huggingface,
    HuggingFaceDatasetAdapter,
    ReplayDatasetAdapter,
)
from torchrl.data.datasets.atari_dqn import AtariDQNExperienceReplay
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.gen_dgrl import GenDGRLExperienceReplay
from torchrl.data.datasets.lerobot import (
    lerobot_columns_to_tensordict,
    LeRobotExperienceReplay,
)
from torchrl.data.datasets.minari_data import (
    _extract_nontensor_fields,
    _NAME_MATCH,
    _patch_nontensor_data_to_stack,
    _preallocate_nontensor_fields,
    MinariExperienceReplay,
)
from torchrl.data.datasets.openml import OpenMLExperienceReplay
from torchrl.data.datasets.openx import OpenXExperienceReplay
from torchrl.data.datasets.roboset import RobosetExperienceReplay
from torchrl.data.datasets.vd4rl import VD4RLExperienceReplay
from torchrl.data.utils import CloudpickleWrapper
from torchrl.data.video import _has_torchcodec
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
_has_datasets = importlib.util.find_spec("datasets") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None

if _has_datasets and _has_pyarrow:
    import datasets
    import pyarrow as pa


class TestHuggingFaceDatasetAdapter:
    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_export_cache_identity_tracks_contents(self, tmp_path):
        data = TensorDict({"obs": torch.arange(5)}, [5])
        source = TensorStorage(data).as_dataset()
        adapter = HuggingFaceDatasetAdapter(batch_size=2)

        def increment(row):
            return {"obs": row["obs"] + 1}

        cache_names = []
        for index in range(3):
            if index == 2:
                data["obs"].add_(10)
            external = adapter.to_external(source)
            path = tmp_path / f"export-{index}"
            external.save_to_disk(path)
            processed = datasets.load_from_disk(path).map(increment)
            assert processed[:]["obs"] == (
                [11, 12, 13, 14, 15] if index == 2 else [1, 2, 3, 4, 5]
            )
            cache_names.append(Path(processed.cache_files[0]["filename"]).name)

        assert cache_names[0] == cache_names[1]
        assert cache_names[1] != cache_names[2]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_batched_columns_are_independent(self, batch_size):
        data = TensorDict(
            {
                "observation": torch.arange(120, dtype=torch.int16).reshape(10, 3, 4),
                ("next", "reward"): torch.arange(10, dtype=torch.float64),
                ("next", "value"): torch.arange(10, dtype=torch.bfloat16),
                ("metadata", "task"): [f"task_{index}" for index in range(10)],
                "done": torch.arange(10) % 2 == 0,
            },
            [10],
        )
        source = Subset(
            TensorStorage(data).as_dataset(), [9, 0, 7, 2, 5, 4, 3, 6, 1, 8]
        )
        adapter = HuggingFaceDatasetAdapter(batch_size=batch_size)
        rng_state = torch.random.get_rng_state()
        external = adapter.to_external(source)
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        data.zero_()
        imported = adapter.from_external(external)
        for index, value in enumerate(source.indices):
            assert imported[index]["observation"].dtype is torch.int16
            assert (
                imported[index]["observation"].tolist()
                == torch.arange(value * 12, (value + 1) * 12).reshape(3, 4).tolist()
            )
            assert imported[index]["next", "reward"].dtype is torch.float64
            assert imported[index]["next", "reward"].item() == value
            assert imported[index]["next", "value"].dtype is torch.float32
            assert imported[index]["next", "value"].item() == value
            assert imported[index]["done"].item() == (value % 2 == 0)
            assert imported[index]["metadata", "task"] == f"task_{value}"

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_export_repeats_compact_non_tensor_data(self, batch_size):
        row_count = 3
        data = TensorDict(
            {
                "observation": torch.arange(row_count),
                # These values are one object shared by the whole TensorDict
                # batch, not sequences of one value per row.
                "same_length_text": NonTensorData("cat", batch_size=[row_count]),
                "different_length_text": NonTensorData("goal", batch_size=[row_count]),
                "none": NonTensorData(None, batch_size=[row_count]),
                "integer": NonTensorData(7, batch_size=[row_count]),
                "float": NonTensorData(1.5, batch_size=[row_count]),
                "boolean": NonTensorData(True, batch_size=[row_count]),
                "sequence": NonTensorData(["left", "right"], batch_size=[row_count]),
                "numpy_array": NonTensorData(
                    np.array([1, 2], dtype=np.int16), batch_size=[row_count]
                ),
                "tensor": NonTensorData(torch.tensor([3, 4]), batch_size=[row_count]),
                "mapping": NonTensorData({"kind": "expert"}, batch_size=[row_count]),
            },
            batch_size=[row_count],
        )
        source = TensorStorage(data).as_dataset()

        external = HuggingFaceDatasetAdapter(batch_size=batch_size).to_external(source)

        assert len(external) == row_count
        assert external[:]["same_length_text"] == ["cat"] * row_count
        assert external[:]["different_length_text"] == ["goal"] * row_count
        assert external[:]["none"] == [None] * row_count
        assert external[:]["integer"] == [7] * row_count
        assert external[:]["float"] == [1.5] * row_count
        assert external[:]["boolean"] == [True] * row_count
        assert external[:]["sequence"] == [["left", "right"]] * row_count
        assert external[:]["numpy_array"] == [[1, 2]] * row_count
        assert external[:]["tensor"] == [[3, 4]] * row_count
        assert [external[index]["mapping"] for index in range(row_count)] == [
            {"kind": "expert"}
        ] * row_count

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_export_non_tensor_stack_and_nested_non_tensor_data(self):
        row_count = 3
        data = TensorDict(
            {
                "observation": torch.arange(row_count),
                "task": ["pick", "pick", "drop"],
                "metadata": TensorDict(
                    {"source": NonTensorData("demonstration", batch_size=[row_count])},
                    batch_size=[row_count],
                ),
            },
            batch_size=[row_count],
        )
        assert isinstance(data.get("task"), NonTensorStack)

        external = HuggingFaceDatasetAdapter().to_external(
            TensorStorage(data).as_dataset()
        )

        assert external[:]["task"] == ["pick", "pick", "drop"]
        assert [external[index]["metadata"] for index in range(row_count)] == [
            {"source": "demonstration"}
        ] * row_count

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_export_repeats_metadata(self):
        row_count = 3
        config = {"dataset": "replay", "version": 2}
        data = TensorDict(
            {
                "observation": torch.arange(row_count),
                "metadata": MetaData(config, batch_size=[row_count]),
            },
            batch_size=[row_count],
        )

        external = HuggingFaceDatasetAdapter().to_external(
            TensorStorage(data).as_dataset()
        )

        assert [external[index]["metadata"] for index in range(row_count)] == [
            config
        ] * row_count

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("nested", [False, True])
    def test_export_rejects_unbatched_tensor(self, nested):
        key = ("next", "lookup") if nested else "lookup"
        data = TensorDict(
            {
                "observation": torch.arange(3),
                key: UnbatchedTensor(torch.tensor([10, 20])),
            },
            batch_size=[3],
        )

        with pytest.raises(ValueError, match="UnbatchedTensor at key") as exc:
            HuggingFaceDatasetAdapter().to_external(TensorStorage(data).as_dataset())
        assert repr(key) in str(exc.value)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("nested", [False, True])
    def test_rejects_columns_changed_between_batches(self, nested):
        source = [
            TensorDict({"a": torch.tensor(0)}, []),
            TensorDict({"b": torch.tensor(1)}, []),
        ]
        if nested:
            source = [TensorDict({"row": item}, []) for item in source]
        adapter = HuggingFaceDatasetAdapter(batch_size=1)
        with pytest.raises(ValueError, match="row 1 has different column keys"):
            adapter.to_external(source)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_variable_columns_across_batches(self):
        source = [TensorDict({"obs": torch.arange(index)}, []) for index in range(4)]
        external = HuggingFaceDatasetAdapter(batch_size=1).to_external(source)
        assert external[:]["obs"] == [[], [0], [0, 1], [0, 1, 2]]
        external = HuggingFaceDatasetAdapter(batch_size=1).to_external(
            [
                TensorDict({"value": NonTensorData(None)}, []),
                TensorDict({"value": NonTensorData("present")}, []),
            ]
        )
        assert external[:]["value"] == [None, "present"]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    @pytest.mark.parametrize("inner_shape", [(2,), (2, 3), (0,)])
    @pytest.mark.parametrize("shared_config", [False, True])
    def test_nested_batch_metadata_fetch_paths(
        self, batch_size, inner_shape, shared_config
    ):
        shape = (4, *inner_shape)
        config = {"source": "replay"}
        configs = (
            MetaData(config, batch_size=shape)
            if shared_config
            else torch.stack(
                [
                    MetaData({"source": f"replay-{index}"}, batch_size=inner_shape)
                    for index in range(4)
                ]
            )
        )
        data = TensorDict(
            {
                "obs": torch.arange(4),
                "agents": TensorDict(
                    {
                        "value": torch.arange(np.prod(shape)).reshape(shape),
                        "task": NonTensorData("goal", batch_size=shape),
                        "config": configs,
                    },
                    shape,
                ),
            },
            [4],
        )
        storage_source = TensorStorage(data).as_dataset()
        adapter = HuggingFaceDatasetAdapter(batch_size=batch_size)
        expected_tasks = np.full(inner_shape, "goal").tolist()
        for source, order in (
            (storage_source, range(4)),
            (list(data.unbind(0)), range(4)),
            (Subset(storage_source, [3, 1, 0, 2]), [3, 1, 0, 2]),
        ):
            external = adapter.to_external(source)
            assert len(external) == 4
            for index, original_index in enumerate(order):
                assert external[index] == {
                    "obs": original_index,
                    "agents": {
                        "value": data[original_index]["agents", "value"].tolist(),
                        "task": expected_tasks,
                        "config": config
                        if shared_config
                        else {"source": f"replay-{original_index}"},
                    },
                }
            assert (
                adapter.to_external(adapter.from_external(external))[:] == external[:]
            )

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        "dtypes",
        [
            (torch.int64, torch.float32),
            (torch.float32, torch.float64),
            (torch.bool, torch.int64),
        ],
    )
    def test_rejects_mixed_tensor_dtypes(self, batch_size, dtypes):
        key = ("next", "value")
        rows = [
            TensorDict({key: torch.tensor(16777217, dtype=dtypes[0])}, []),
            TensorDict({key: torch.tensor(0.5, dtype=dtypes[1])}, []),
        ]
        adapter = HuggingFaceDatasetAdapter(batch_size=batch_size)
        storage = ListStorage(2)
        storage.set(range(2), rows)
        for source in (rows, storage.as_dataset()):
            with pytest.raises(
                ValueError, match="consistent tensor dtype per key"
            ) as exc:
                adapter.to_external(source)
            assert repr(key) in str(exc.value)
        # A failed export must not retain dtype state for the next source.
        external = adapter.to_external(
            [TensorDict({key: torch.tensor(16777217, dtype=torch.int64)}, [])]
        )
        assert external[0]["next"]["value"] == 16777217

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    @pytest.mark.parametrize("observation", [False, True])
    def test_rejects_empty_nested_structures(self, batch_size, observation):
        rows = [
            TensorDict({("metadata", "empty"): TensorDict({}, [])}, [])
            for _ in range(3)
        ]
        if observation:
            for index, row in enumerate(rows):
                row["obs"] = torch.tensor(index)
        with pytest.raises(ValueError, match="empty nested TensorDicts") as exc:
            HuggingFaceDatasetAdapter(batch_size=batch_size).to_external(rows)
        assert repr(("metadata", "empty")) in str(exc.value)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize(
        "shape", [(), (0,), (2, 0), (2, 3), (0, 3), (2, 0, 3), (0, 0)]
    )
    def test_empty_tensor_dimensions(self, shape, tmp_path):
        rows = [
            TensorDict({("next", "obs"): torch.zeros(shape, dtype=torch.float64)}, [])
            for _ in range(2)
        ]
        adapter = HuggingFaceDatasetAdapter()
        if 0 in shape[:-1]:
            with pytest.raises(ValueError, match="empty inner dimension") as exc:
                adapter.to_external(rows)
            assert repr(("next", "obs")) in str(exc.value)
            return
        external = adapter.to_external(rows)
        path = tmp_path / "empty.parquet"
        external.to_parquet(path)
        reloaded = datasets.load_dataset("parquet", data_files=str(path), split="train")
        for source in (external, reloaded):
            imported = adapter.from_external(source)
            assert len(imported) == 2
            assert_allclose_td(imported[0], rows[0])
            assert imported[0]["next", "obs"].shape == shape
            assert imported[0]["next", "obs"].dtype is torch.float64

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("dtype", ["bool", "float32", "int16"])
    @pytest.mark.parametrize("nested_lists", [False, True])
    def test_nullable_numeric_sequences(self, dtype, nested_lists, tmp_path):
        first, last = (True, False) if dtype == "bool" else (1, 3)
        missing, present = [first, None, last], [first, last, first]
        feature = datasets.Sequence(datasets.Value(dtype))
        if nested_lists:
            missing, present = [[first, None], None, [last, first]], [
                [first, last],
                [last, first],
            ]
            feature = datasets.Sequence(feature)
        values = [missing, present, None]
        external = datasets.Dataset.from_dict(
            {"next": [{"value": value} for value in values]},
            features=datasets.Features({"next": {"value": feature}}),
        )
        path = tmp_path / "nullable.parquet"
        external.to_parquet(path)
        reloaded = datasets.load_dataset("parquet", data_files=str(path), split="train")
        adapter = HuggingFaceDatasetAdapter(batch_size=1)
        for source in (external, reloaded):
            imported = adapter.from_external(source)
            assert isinstance(imported[0].get(("next", "value")), NonTensorData)
            assert imported[0]["next", "value"] == missing
            assert imported[1]["next", "value"].tolist() == present
            assert imported[1]["next", "value"].dtype == getattr(torch, dtype)
            assert imported[2]["next", "value"] is None
            assert adapter.to_external(imported)[:] == external[:]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize(
        "dtype, values",
        [
            ("bool", [False, True]),
            ("uint8", [0, 255]),
            ("uint64", [0, 2**63 + 1, 2**64 - 1]),
            ("float32", [1.25, float("nan"), float("inf"), -0.0]),
            ("float64", [1.123456789012345, float("nan"), -float("inf"), -0.0]),
        ],
    )
    def test_import_numeric_arrays_with_and_without_nulls(self, dtype, values):
        dense = [[values, values], [values, values]]
        missing_leaf = [[values, values], [values, [*values[:-1], None]]]
        missing_container = [None, [values, values]]
        feature = datasets.Value(dtype)
        for _ in range(3):
            feature = datasets.Sequence(feature)
        external = datasets.Dataset(
            pa.Table.from_pydict(
                {"value": [dense, missing_leaf, missing_container]},
                schema=datasets.Features({"value": feature}).arrow_schema,
            )
        )

        imported = HuggingFaceDatasetAdapter().from_external(external)
        tensor = imported[0]["value"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == getattr(torch, dtype)
        np.testing.assert_equal(tensor.numpy(), np.asarray(dense, dtype=dtype))
        for index, expected in enumerate((missing_leaf, missing_container), start=1):
            assert isinstance(imported[index].get("value"), NonTensorData)
            np.testing.assert_equal(imported[index]["value"], expected)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_nullable_float_preserves_nan(self):
        external = datasets.Dataset.from_dict(
            {"value": [[None, float("nan"), 1.25]]},
            features=datasets.Features(
                {"value": datasets.Sequence(datasets.Value("float64"))}
            ),
        )
        adapter = HuggingFaceDatasetAdapter()
        for source in (external, adapter.to_external(adapter.from_external(external))):
            value = adapter.from_external(source)[0]["value"]
            assert value[0] is None
            assert np.isnan(value[1])
            assert value[2] == 1.25

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("in_sequence", [False, True])
    @pytest.mark.parametrize(
        "shape, value, expected_shape",
        [
            ((0, 2), [], (0, 2)),
            ((2, 0, 3), [[], []], (2, 0, 3)),
            ((None, 2), [], (0, 2)),
            ((None, 0, 3), [[], []], (2, 0, 3)),
            ((2, 2), [[1, 2], [3, 4]], (2, 2)),
        ],
    )
    def test_import_declared_array_shapes(
        self, shape, value, expected_shape, in_sequence, tmp_path
    ):
        feature = getattr(datasets, f"Array{len(shape)}D")(shape, "float64")
        if in_sequence:
            feature = datasets.Sequence(feature)
            value = [value]
            expected_shape = (1, *expected_shape)
        external = datasets.Dataset.from_dict(
            {"next": [{"value": value}]},
            features=datasets.Features({"next": {"value": feature}}),
        )
        external.save_to_disk(tmp_path / "arrays")
        for source in (external, datasets.load_from_disk(tmp_path / "arrays")):
            result = HuggingFaceDatasetAdapter().from_external(source)[0][
                "next", "value"
            ]
            assert result.shape == expected_shape
            assert result.dtype is torch.float64
            assert result.tolist() == value

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_import_array_beneath_empty_sequence(self, dynamic):
        feature = datasets.Sequence(
            datasets.Array2D((None if dynamic else 2, 3), "int16")
        )
        external = datasets.Dataset.from_dict(
            {"value": [[]]}, features=datasets.Features({"value": feature})
        )
        imported = HuggingFaceDatasetAdapter().from_external(external)
        if dynamic:
            with pytest.raises(ValueError, match="dynamic.*empty sequence"):
                imported[0]
        else:
            assert imported[0]["value"].shape == (0, 2, 3)
            assert imported[0]["value"].dtype is torch.int16

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("length", [-1, 4])
    def test_import_array_beneath_empty_nested_sequence(self, length):
        feature = datasets.Sequence(
            datasets.Sequence(datasets.Array2D((2, 3), "int16"), length=length)
        )
        external = datasets.Dataset.from_dict(
            {"value": [[]]}, features=datasets.Features({"value": feature})
        )
        imported = HuggingFaceDatasetAdapter().from_external(external)
        if length < 0:
            with pytest.raises(ValueError, match="dynamic.*empty sequence"):
                imported[0]
        else:
            assert imported[0]["value"].shape == (0, 4, 2, 3)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_nullable_array_features(self):
        value = [[1, None], [2, 3]]
        feature = datasets.Array2D((2, 2), "float32")
        adapter = HuggingFaceDatasetAdapter()
        external = datasets.Dataset.from_dict(
            {"value": [value]}, features=datasets.Features({"value": feature})
        )
        with pytest.raises(ValueError) as original_error:
            external[0]
        with pytest.raises(ValueError, match="nullable ArrayXD") as error:
            adapter.from_external(external)[0]
        assert str(original_error.value) in str(error.value)
        external = datasets.Dataset.from_dict(
            {"value": [[value]]},
            features=datasets.Features({"value": datasets.Sequence(feature)}),
        )
        assert adapter.from_external(external)[0]["value"] == [value]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="requires float8")
    def test_float8_error_reports_nested_key(self):
        value = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn)
        with pytest.raises(TypeError, match="Cannot convert tensor") as exc:
            HuggingFaceDatasetAdapter().to_external(
                [TensorDict({("next", "obs"): value}, [])]
            )
        assert repr(("next", "obs")) in str(exc.value)
        assert str(value.dtype) in str(exc.value)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("format_type", ["torch", "numpy", "custom", "columns"])
    def test_import_reads_unformatted_values(self, format_type):
        values = [1.123456789012345, 2.987654321098765]
        external = datasets.Dataset.from_dict(
            {"value": values, "count": [7, 8]},
            features=datasets.Features(
                {"value": datasets.Value("float64"), "count": datasets.Value("int16")}
            ),
        ).select([1, 0])
        if format_type == "custom":
            external.set_transform(
                lambda batch: {"value": [v + 0.5 for v in batch["value"]]}
            )
        elif format_type == "columns":
            external.set_format("torch", columns=["value"])
        else:
            external.set_format(format_type)
        original_format = external.format
        imported = HuggingFaceDatasetAdapter().from_external(external)
        for index, original_index in enumerate([1, 0]):
            assert imported[index]["value"].item() == values[original_index]
            assert imported[index]["value"].dtype is torch.float64
            assert imported[index]["count"].item() == 7 + original_index
            assert imported[index]["count"].dtype is torch.int16
        assert external.format == original_format
        if format_type == "custom":
            assert external[0]["value"] == values[1] + 0.5
        elif format_type == "columns":
            assert set(external[0]) == {"value"}

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("payload_kind", ["numpy", "tensor"])
    def test_non_tensor_payloads_are_snapshots(self, payload_kind):
        payload = np.arange(4, dtype=np.int16).reshape(2, 2)
        if payload_kind == "tensor":
            payload = torch.from_numpy(payload)
        data = TensorDict({"payload": NonTensorData(payload, batch_size=[3])}, [3])
        external = HuggingFaceDatasetAdapter().to_external(
            TensorStorage(data).as_dataset()
        )
        payload[...] = 99
        assert external[:]["payload"] == [[[0, 1], [2, 3]]] * 3

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("kind", ["complex", "sparse", "quantized"])
    def test_rejects_unsupported_tensors(self, kind):
        value = torch.tensor([1.0, 2.0])
        if kind == "complex":
            value = torch.tensor([1 + 2j])
        elif kind == "sparse":
            value = value.to_sparse()
        else:
            value = torch.quantize_per_tensor(value, 0.1, 0, torch.qint8)
        with pytest.raises(TypeError, match="tensor layout or dtype"):
            HuggingFaceDatasetAdapter().to_external([TensorDict({"obs": value}, [])])

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_shuffled_noncontiguous_round_trip(self, batch_size, seed):
        dtype = (torch.float64, torch.int16, torch.bool, torch.bfloat16)[seed]
        values = torch.arange(19 * 6).reshape(19, 2, 3).transpose(1, 2).to(dtype)
        if dtype.is_floating_point:
            values.requires_grad_()
        expected = values.detach().clone()
        tasks = [None if index % 3 == 0 else f"task-{index % 2}" for index in range(19)]
        data = TensorDict(
            {
                ("next", "obs"): values,
                "task": tasks,
                "tags": NonTensorData([1, None, 3], batch_size=[19]),
            },
            [19],
        )
        indices = torch.randperm(
            19, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sources = [
            Subset(TensorStorage(data).as_dataset(), indices),
            [data[index] for index in indices],
        ]
        adapter = HuggingFaceDatasetAdapter(batch_size=batch_size)
        rng_state = torch.random.get_rng_state()
        exports = [adapter.to_external(source) for source in sources]
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        with torch.no_grad():
            values.zero_()
        for external in exports:
            imported = adapter.from_external(external)
            assert len(imported) == 19
            for index, original_index in enumerate(indices):
                row = imported[index]
                assert row["next", "obs"].tolist() == expected[original_index].tolist()
                assert row["next", "obs"].dtype == (
                    torch.float32 if dtype is torch.bfloat16 else dtype
                )
                assert row["task"] == tasks[original_index]
                assert row["tags"] == [1, None, 3]

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="positive"):
            HuggingFaceDatasetAdapter(batch_size=batch_size)

    @pytest.mark.parametrize("batch_size", [True, 1.5, "2", None])
    def test_invalid_batch_size_type(self, batch_size):
        with pytest.raises(TypeError, match="positive integer"):
            HuggingFaceDatasetAdapter(batch_size=batch_size)

    def test_missing_dependency(self, monkeypatch):
        monkeypatch.setattr(huggingface, "_has_datasets", False)
        with pytest.raises(ImportError, match="requires the `datasets` package"):
            huggingface.HuggingFaceDatasetAdapter()

        monkeypatch.setattr(huggingface, "_has_datasets", True)
        monkeypatch.setattr(huggingface, "_has_pyarrow", False)
        with pytest.raises(ImportError, match="requires the `pyarrow` package"):
            huggingface.HuggingFaceDatasetAdapter()

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_docstring_examples(self):
        runner = doctest.DocTestRunner()
        example = doctest.DocTestParser().get_doctest(
            HuggingFaceDatasetAdapter.__doc__, {}, "HuggingFaceDatasetAdapter", None, 0
        )
        assert runner.run(example).failed == 0

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_storage_parquet_round_trip(self, tmp_path):
        rows = [
            TensorDict(
                {
                    "observation": torch.tensor(
                        [index, index + 0.5], dtype=torch.float64
                    ),
                    "action": torch.tensor(index, dtype=torch.int16),
                    "done": torch.tensor(index == 1),
                    "next": TensorDict(
                        {"reward": torch.tensor(index / 2, dtype=torch.float32)},
                        batch_size=[],
                    ),
                    "task": NonTensorData("pick" if index == 0 else "place"),
                },
                batch_size=[],
            )
            for index in range(2)
        ]
        storage = ListStorage(2)
        storage.set(range(2), rows)
        adapter: ReplayDatasetAdapter[datasets.Dataset] = HuggingFaceDatasetAdapter()

        hf_dataset = adapter.to_external(storage.as_dataset())
        direct = adapter.from_external(hf_dataset)

        path = tmp_path / "replay.parquet"
        hf_dataset.to_parquet(path)
        reloaded = datasets.load_dataset("parquet", data_files=str(path), split="train")
        imported = adapter.from_external(reloaded)

        assert isinstance(hf_dataset, datasets.Dataset)
        assert len(imported) == len(rows)
        for index, expected in enumerate(rows):
            assert_allclose_td(direct[index], expected)
            assert_allclose_td(imported[index], expected)
            assert imported[index]["observation"].dtype is torch.float64
            assert imported[index]["action"].dtype is torch.int16
            assert imported[index]["next", "reward"].dtype is torch.float32
        batch = next(
            iter(
                DataLoader(
                    imported,
                    batch_size=2,
                    collate_fn=tensordict_collate,
                )
            )
        )
        assert batch.batch_size == torch.Size([2])
        assert list(batch["task"]) == ["pick", "place"]
        storage[0]["observation"].fill_(99)
        assert direct[0]["observation"].tolist() == [0, 0.5]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_rejects_batched_items(self):
        storage = ListStorage(1)
        storage.set(0, TensorDict({"observation": torch.arange(6).reshape(2, 3)}, [2]))
        source = storage.as_dataset()
        with pytest.raises(ValueError, match="unbatched TensorDict"):
            HuggingFaceDatasetAdapter().to_external(source)

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("picklable", [False, True])
    def test_materializes_current_backing_data(self, tmp_path, picklable):
        class FileRows(Dataset[TensorDictBase]):
            def __init__(self, path, picklable):
                self.path, self.picklable = path, picklable

            def __len__(self):
                return 2

            def __getitem__(self, index):
                value = int(self.path.read_text().splitlines()[index])
                return TensorDict({"obs": torch.tensor(value)}, [])

            def __getstate__(self):
                if not self.picklable:
                    raise TypeError("Source does not support serialization.")
                return self.__dict__

        path = tmp_path / "rows.txt"
        path.write_text("1\n2\n")
        source = FileRows(path, picklable)
        adapter = HuggingFaceDatasetAdapter()

        first = adapter.to_external(source)
        path.write_text("1\n99\n")
        second = adapter.to_external(source)

        assert first[:]["obs"] == [1, 2]
        assert second[:]["obs"] == [1, 99]

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    @pytest.mark.parametrize("nested", [False, True])
    def test_rejects_inconsistent_columns(self, nested):
        rows = [
            TensorDict({"a": torch.tensor(0)}, []),
            TensorDict({"a": torch.tensor(1), "b": torch.tensor(7)}, []),
        ]
        if nested:
            rows = [TensorDict({"row": item}, []) for item in rows]
        storage = ListStorage(2)
        storage.set(range(2), rows)
        adapter = HuggingFaceDatasetAdapter()

        with pytest.raises(ValueError, match="same keys and tensor shapes"):
            adapter.to_external(storage.as_dataset())

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_empty_and_invalid_inputs(self):
        storage = ListStorage(1)
        adapter = HuggingFaceDatasetAdapter()
        empty = adapter.to_external(storage.as_dataset())
        assert len(empty) == len(adapter.from_external(empty)) == 0
        storage.set(0, TensorDict({"obs": torch.tensor(0)}, []))
        source = storage.as_dataset()

        with pytest.raises(TypeError, match="expects TensorDict items"):
            adapter.to_external([torch.tensor(0)])
        with pytest.raises(ValueError, match="nonempty TensorDict"):
            adapter.to_external([TensorDict({}, [])])

        external = adapter.to_external(source)
        with pytest.raises(TypeError, match="Select a split"):
            adapter.from_external(datasets.DatasetDict(train=external))

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets and pyarrow"
    )
    def test_non_numpy_feature(self):
        external = datasets.Dataset.from_dict(
            {"price": [Decimal("1.25")]},
            features=datasets.Features({"price": datasets.Value("decimal128(3, 2)")}),
        )

        imported = HuggingFaceDatasetAdapter().from_external(external)

        assert imported[0]["price"] == Decimal("1.25")

    @pytest.mark.skipif(
        not (_has_datasets and _has_pyarrow), reason="requires datasets"
    )
    @pytest.mark.parametrize("storage_type", [ListStorage, LazyTensorStorage])
    @pytest.mark.parametrize("shared", [False, True])
    def test_replay_buffer_episodic_export(self, storage_type, shared):
        rb = ReplayBuffer(storage=storage_type(4), shared=shared)
        rb.extend(
            TensorDict(
                {
                    "obs": torch.tensor([11, 1, 10, 0]),
                    ("collector", "traj_ids"): torch.tensor([1, 0, 1, 0]),
                    "step": torch.tensor([1, 1, 0, 0]),
                },
                [4],
            )
        )
        adapter = HuggingFaceDatasetAdapter()
        external = rb.export(
            adapter, trajectory_key=("collector", "traj_ids"), step_key="step"
        )
        rb.empty()
        rb.extend(
            TensorDict(
                {
                    "obs": torch.tensor([99]),
                    ("collector", "traj_ids"): torch.tensor([9]),
                    "step": torch.tensor([0]),
                },
                [1],
            )
        )
        imported = adapter.from_external(external)
        assert [imported[i]["obs"].item() for i in range(len(imported))] == [
            0,
            1,
            10,
            11,
        ]
        assert [
            imported[i]["collector", "traj_ids"].item() for i in range(len(imported))
        ] == [0, 0, 1, 1]


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
        import gym

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


def _minari_like_episode(missions: list[bytes], n_actions: int) -> TensorDict:
    """Build one H5-style Minari episode with a NonTensorData mission array."""
    n_obs = n_actions + 1
    if len(missions) != n_obs:
        raise ValueError("missions must cover the reset observation plus each step")
    return TensorDict(
        {
            "actions": torch.zeros(n_actions, 1),
            "rewards": torch.zeros(n_actions),
            "terminations": torch.tensor(
                [False] * (n_actions - 1) + [True], dtype=torch.bool
            ),
            "truncations": torch.zeros(n_actions, dtype=torch.bool),
            "observations": TensorDict(
                {
                    "image": torch.zeros(n_obs, 2, 2, 3, dtype=torch.uint8),
                    "mission": NonTensorData(
                        np.array(missions, dtype=object),
                        batch_size=[n_obs],
                    ),
                },
                batch_size=[n_obs],
            ),
        }
    )


def test_patch_nontensor_data_to_stack_indexes_one_mission_per_step():
    """``NonTensorData`` of a mission array must become a per-step stack."""
    episode = _minari_like_episode(
        [b"pick up a blue box", b"pick up a green key"], n_actions=1
    )
    patched = _patch_nontensor_data_to_stack(episode.clone())
    missions = patched.get(("observations", "mission"))
    assert missions[0].data == b"pick up a blue box"
    assert missions[1].data == b"pick up a green key"


def test_preallocate_nontensor_fields_does_not_alias_next_mission():
    """Current and next mission stacks must be distinct objects (#3105).

    The download fill updates the TED current slice, then the next slice.
    A shared NonTensorStack makes next track current, so missions never shift.
    """
    missions = [b"pick up a blue box", b"pick up a green key", b"open the red door"]
    episode = _minari_like_episode(missions, n_actions=2)
    dest = TensorDict({}, batch_size=[2])
    _preallocate_nontensor_fields(dest, episode, total_steps=2, name_map=_NAME_MATCH)
    current = dest.get(("observation", "mission"))
    nxt = dest.get(("next", "observation", "mission"))
    assert current is not nxt

    patched = _patch_nontensor_data_to_stack(episode.clone())
    dest.get("observation").update_(
        _extract_nontensor_fields(patched.get("observations")[:-1].clone())
    )
    dest.get(("next", "observation")).update_(
        _extract_nontensor_fields(patched.get("observations")[1:].clone())
    )
    assert dest[0]["observation", "mission"] == b"pick up a blue box"
    assert dest[0]["next", "observation", "mission"] == b"pick up a green key"
    assert dest[1]["observation", "mission"] == b"pick up a green key"
    assert dest[1]["next", "observation", "mission"] == b"open the red door"


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
            import gymnasium
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

    @pytest.mark.skipif(
        not _has_minari or not _has_gymnasium, reason="Minari or Gym not available"
    )
    def test_download_load_dataset_path(self, tmp_path, monkeypatch):
        """Regression test for the ``MINARI_DATASETS_PATH`` handling on download.

        ``_download_and_preproc`` downloads the dataset into a temporary directory
        and then calls ``minari.load_dataset`` to read its metadata. Minari
        resolves dataset locations through ``MINARI_DATASETS_PATH``; if that
        variable is restored to a pre-existing value before the ``load_dataset``
        call, Minari looks in the wrong directory and raises ``FileNotFoundError``.

        Here we simulate a remote download by copying a locally-built dataset into
        whatever directory torchrl points Minari at, while ``MINARI_DATASETS_PATH``
        is set to an unrelated, empty cache. This used to fail before the fix.
        """
        import gymnasium
        import minari
        from minari import DataCollector

        dataset_id = "cartpole/test-download-load-v1"

        source_dir = tmp_path / "source"
        monkeypatch.setenv("MINARI_DATASETS_PATH", str(source_dir))
        env = DataCollector(gymnasium.make("CartPole-v1"), record_infos=True)
        for _ in range(5):
            env.reset(seed=123)
            while True:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
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

        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        monkeypatch.setenv("MINARI_DATASETS_PATH", str(empty_cache))

        def fake_download_dataset(dataset_id, *args, **kwargs):
            dest = os.environ["MINARI_DATASETS_PATH"]
            shutil.copytree(source_dir, dest, dirs_exist_ok=True)

        monkeypatch.setattr(minari, "download_dataset", fake_download_dataset)

        data = MinariExperienceReplay(
            dataset_id=dataset_id,
            batch_size=32,
            root=str(tmp_path / "torchrl_root"),
            download=True,
        )
        sample = data.sample()
        assert sample.shape[0] == 32


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
    @pytest.fixture(autouse=True)
    def _patch_openml_data(self, monkeypatch, tmp_path):
        """Use deterministic local data instead of querying OpenML in CI tests."""

        def _get_data(cls, dataset_name):
            dataset_seed = sum(map(ord, dataset_name))
            generator = torch.Generator().manual_seed(dataset_seed)
            num_rows = 5000
            num_features = 8
            num_classes = 3
            x = torch.randn(
                num_rows, num_features, dtype=torch.float64, generator=generator
            )
            y = torch.randint(num_classes, (num_rows,), generator=generator)
            return TensorDict({"X": x, "y": y}, [num_rows])

        monkeypatch.setattr(OpenMLExperienceReplay, "_get_data", classmethod(_get_data))
        monkeypatch.setattr(
            "torchrl.data.datasets.openml._get_root_dir",
            lambda name: tmp_path / name,
        )

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


class TestLeRobotExperienceReplay:
    @staticmethod
    def _columns(n_ep=2, ep_len=4, state_dim=7, action_dim=7):
        n = n_ep * ep_len
        return {
            "observation.state": torch.randn(n, state_dim),
            "observation.images.top": torch.randint(
                0, 255, (n, 3, 8, 8), dtype=torch.uint8
            ),
            "action": torch.randn(n, action_dim),
            "episode_index": torch.arange(n_ep).repeat_interleave(ep_len),
            "frame_index": torch.arange(n) % ep_len,
            "task": ["pick the cube"] * (n // 2) + ["place the cube"] * (n - n // 2),
            "next.reward": torch.zeros(n, 1),
        }

    def test_columns_to_tensordict(self):
        td = lerobot_columns_to_tensordict(self._columns())
        assert td["observation", "state"].shape == torch.Size([8, 7])
        assert td["observation", "image", "top"].shape == torch.Size([8, 3, 8, 8])
        assert td["action"].shape == torch.Size([8, 7])
        assert td["episode"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert td["next", "reward"].shape == torch.Size([8, 1])
        assert td.get("language_instruction").tolist()[0] == "pick the cube"

    def test_multi_camera_mapping(self):
        cols = {
            "observation.images.top": torch.zeros(2, 3, 4, 4),
            "observation.images.wrist": torch.zeros(2, 3, 4, 4),
            "action": torch.zeros(2, 7),
            "episode_index": torch.zeros(2, dtype=torch.long),
        }
        td = lerobot_columns_to_tensordict(cols)
        keys = set(td.keys(include_nested=True, leaves_only=True))
        assert ("observation", "image", "top") in keys
        assert ("observation", "image", "wrist") in keys

    def test_custom_key_map(self):
        cols = {
            "act": torch.zeros(3, 2),
            "episode_index": torch.zeros(3, dtype=torch.long),
        }
        td = lerobot_columns_to_tensordict(cols, key_map={"act": "action"})
        assert td["action"].shape == torch.Size([3, 2])

    def test_from_columns_sample_windows(self):
        rb = LeRobotExperienceReplay.from_columns(
            self._columns(n_ep=2, ep_len=4), slice_len=4, batch_size=8
        )
        sample = rb.sample()
        assert sample.numel() == 8
        windows = sample.reshape(2, 4)
        # slices never cross trajectory boundaries
        for i in range(2):
            ep = windows["episode"][i]
            assert (ep == ep[0]).all()
        assert sample.get("language_instruction") is not None

    def test_download_requires_hf_packages(self, tmp_path, monkeypatch):
        import torchrl.data.datasets.lerobot as lerobot_module

        monkeypatch.setattr(lerobot_module, "_has_datasets", False)
        with pytest.raises(ImportError, match="datasets"):
            LeRobotExperienceReplay("fake/repo", root=str(tmp_path), download=True)

    def test_next_keys_get_trailing_singleton(self):
        # TED convention: per-step signals under "next" carry a trailing
        # singleton dim (SliceSampler combines them with its own [batch, 1]
        # truncated flag), while index-like columns stay 1-D.
        td = lerobot_columns_to_tensordict(
            {
                "next.done": torch.zeros(4, dtype=torch.bool),
                "next.reward": torch.zeros(4),
                "episode_index": torch.zeros(4, dtype=torch.long),
            }
        )
        assert td["next", "done"].shape == torch.Size([4, 1])
        assert td["next", "reward"].shape == torch.Size([4, 1])
        assert td["episode"].shape == torch.Size([4])

    @staticmethod
    def _snapshot(major_version, info, episodes):
        from torchrl.data.datasets.lerobot import _LeRobotSnapshot

        snap = _LeRobotSnapshot.__new__(_LeRobotSnapshot)
        snap.root = Path("/data")
        snap.info = info
        snap.fps = float(info.get("fps", 10))
        snap.major_version = major_version
        snap.episodes = episodes
        return snap

    def test_video_segments_v3_grouping(self):
        # v3: episodes are concatenated back-to-back inside shared files; the
        # segments group consecutive episodes per file and sum their lengths
        snap = self._snapshot(
            3,
            {
                "fps": 10,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            },
            [
                {
                    "episode_index": ep,
                    "length": length,
                    "videos/cam/chunk_index": 0,
                    "videos/cam/file_index": file,
                    "videos/cam/from_timestamp": start,
                }
                for ep, (length, file, start) in enumerate(
                    [(10, 0, 0.0), (20, 0, 1.0), (5, 1, 0.0)]
                )
            ],
        )
        paths, counts = snap.video_segments("cam")
        assert counts == [30, 5]
        assert paths == [
            str(Path("/data/videos/cam/chunk-000/file-000.mp4")),
            str(Path("/data/videos/cam/chunk-000/file-001.mp4")),
        ]

    def test_rehydrate_next_obs(self):
        columns = {
            "observation.state": torch.arange(8, dtype=torch.float32).view(8, 1),
            "action": torch.zeros(8, 2),
            "episode_index": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "frame_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            "task": ["t"] * 8,
        }
        rb = LeRobotExperienceReplay.from_columns(
            columns, slice_len=4, batch_size=8, rehydrate=True
        )
        # the invariant must hold on every draw, whichever episodes the
        # sampler picks: a position mirrors its successor iff the successor is
        # the true next step (same episode AND consecutive frame counter)
        for _ in range(20):
            sample = rb.sample()
            state = sample["observation", "state"].squeeze(-1)
            next_state = sample["next", "observation", "state"].squeeze(-1)
            episode = sample["episode"]
            frame = sample["frame"]
            consecutive = (episode[1:] == episode[:-1]) & (frame[1:] == frame[:-1] + 1)
            assert (next_state[:-1][consecutive] == state[1:][consecutive]).all()
            assert next_state[:-1][~consecutive].isnan().all()
            assert next_state[-1].isnan()

    def test_rehydrate_splice_is_filled(self):
        # a single episode guarantees back-to-back slices of the same episode:
        # the splice position must be filled (frame counter check), not given
        # the first row of the following slice
        columns = {
            "observation.state": torch.arange(8, dtype=torch.float32).view(8, 1),
            "episode_index": torch.zeros(8, dtype=torch.long),
            "frame_index": torch.arange(8),
            "task": ["t"] * 8,
        }
        rb = LeRobotExperienceReplay.from_columns(
            columns, slice_len=4, batch_size=8, rehydrate=True
        )
        for _ in range(20):
            sample = rb.sample()
            state = sample["observation", "state"].squeeze(-1)
            next_state = sample["next", "observation", "state"].squeeze(-1)
            frame = sample["frame"]
            splice = frame[4] != frame[3] + 1
            if splice:
                assert next_state[3].isnan()
            else:
                assert next_state[3] == state[4]
            assert next_state[7].isnan()

    def test_rehydrate_requires_episode(self):
        columns = {
            "observation.state": torch.zeros(4, 1),
            "task": ["t"] * 4,
        }
        with pytest.raises(ValueError, match="episode"):
            LeRobotExperienceReplay.from_columns(columns, batch_size=4, rehydrate=True)

    def test_rehydrate_integer_fill(self):
        # integer leaves (e.g. uint8 frames) cannot hold NaN: they are
        # zero-filled at slice boundaries instead
        columns = {
            "observation.state": torch.zeros(8, 1),
            "observation.images.cam": torch.arange(1, 9, dtype=torch.uint8)
            .view(8, 1, 1, 1)
            .expand(8, 3, 2, 2)
            .clone(),
            "episode_index": torch.zeros(8, dtype=torch.long),
            "frame_index": torch.arange(8),
            "task": ["t"] * 8,
        }
        rb = LeRobotExperienceReplay.from_columns(
            columns, slice_len=8, batch_size=8, rehydrate=True
        )
        sample = rb.sample()
        img = sample["observation", "image", "cam"]
        next_img = sample["next", "observation", "image", "cam"]
        assert next_img.dtype == torch.uint8
        assert (next_img[:-1] == img[1:]).all()
        assert (next_img[-1] == 0).all()
        assert sample["next", "observation", "state"][-1].isnan().all()

    def test_rehydrate_skips_undecoded_refs(self):
        # an undecoded video reference cannot be shifted: it is skipped (with
        # a warning) while the tensor observations are still re-hydrated
        ref = VideoClipRef.from_files(["/tmp/fake.mp4"], num_frames_per_file=[8])
        columns = {
            "observation.state": torch.arange(8, dtype=torch.float32).view(8, 1),
            "observation.images.cam": ref,
            "episode_index": torch.zeros(8, dtype=torch.long),
            "frame_index": torch.arange(8),
            "task": ["t"] * 8,
        }
        rb = LeRobotExperienceReplay.from_columns(
            columns, slice_len=8, batch_size=8, rehydrate=True, decode_video=False
        )
        sample = rb.sample()
        assert ("next", "observation", "state") in sample.keys(True, True)
        assert sample.get(("next", "observation", "image"), None) is None

    def test_video_segments_v2_per_episode(self):
        # v2.x: one video file per episode, chunked by episode index
        snap = self._snapshot(
            2,
            {
                "fps": 10,
                "chunks_size": 2,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            },
            [
                {"episode_index": 0, "length": 3},
                {"episode_index": 1, "length": 4},
                {"episode_index": 2, "length": 5},
            ],
        )
        paths, counts = snap.video_segments("cam")
        assert counts == [3, 4, 5]
        assert paths[0] == str(Path("/data/videos/chunk-000/cam/episode_000000.mp4"))
        assert paths[2] == str(Path("/data/videos/chunk-001/cam/episode_000002.mp4"))

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found"):
            LeRobotExperienceReplay("fake/repo", root=str(tmp_path), download=False)

    def test_mixed_list_column_raises(self):
        with pytest.raises(ValueError, match="mixes strings"):
            lerobot_columns_to_tensordict(
                {"x": ["a", 1, "c"], "episode_index": [0, 0, 0]}
            )

    def test_from_columns_is_disk_free(self, tmp_path):
        target = tmp_path / "should_not_be_created"
        LeRobotExperienceReplay.from_columns(
            self._columns(), slice_len=4, batch_size=8, root=str(target)
        )
        assert not target.exists()

    def test_video_clip_ref_column_passthrough(self):
        # A VideoClipRef column flows through to the canonical image key and, with
        # decode_video=False, is sampled lazily (no decoding, no torchcodec needed).
        n = 8
        cols = {
            "observation.images.top": VideoClipRef(
                "dummy.mp4", frame_index=torch.arange(n)
            ),
            "action": torch.randn(n, 7),
            "episode_index": torch.zeros(n, dtype=torch.long),
        }
        td = lerobot_columns_to_tensordict(cols)
        assert isinstance(td.get(("observation", "image", "top")), VideoClipRef)
        rb = LeRobotExperienceReplay.from_columns(
            cols, slice_len=4, batch_size=8, decode_video=False
        )
        sample = rb.sample()
        ref = sample.get(("observation", "image", "top"))
        assert isinstance(ref, VideoClipRef)
        assert ref.shape == torch.Size([8])

    @pytest.mark.skipif(not _has_torchcodec, reason="requires torchcodec")
    def test_video_decode_on_sample(self, tmp_path):
        # End-to-end: a synthetic MP4 is referenced lazily, and decode_video=True
        # (default) decodes the sampled frames on rb.sample().
        from torchcodec.encoders import VideoEncoder

        n, h, w = 8, 16, 16
        frames = torch.arange(n, dtype=torch.uint8).reshape(n, 1, 1, 1)
        frames = frames.expand(n, 3, h, w).contiguous()
        path = str(tmp_path / "clip.mp4")
        VideoEncoder(frames=frames, frame_rate=10).to_file(path)
        cols = {
            "observation.images.top": VideoClipRef.from_file(path),
            "action": torch.randn(n, 7),
            "episode_index": torch.zeros(n, dtype=torch.long),
        }
        rb = LeRobotExperienceReplay.from_columns(cols, slice_len=4, batch_size=8)
        sample = rb.sample()
        decoded = sample.get(("observation", "image", "top"))
        assert torch.is_tensor(decoded)
        assert decoded.shape == torch.Size([8, 3, h, w])
        assert decoded.dtype == torch.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
