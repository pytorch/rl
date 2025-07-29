# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import pickle

import pytest
import torch


try:
    from safetensors.torch import save
except ImportError:
    save = None


class TestCompressedStorageBenchmark:
    """Benchmark tests for CompressedListStorage."""

    @staticmethod
    def make_compressible_mock_data(num_experiences: int, device=None) -> dict:
        """Easily compressible data for testing."""
        if device is None:
            device = torch.device("cpu")

        return {
            "observations": torch.zeros(
                (num_experiences, 4, 84, 84),
                dtype=torch.uint8,
                device=device,
            ),
            "actions": torch.zeros((num_experiences,), device=device),
            "rewards": torch.zeros((num_experiences,), device=device),
            "next_observations": torch.zeros(
                (num_experiences, 4, 84, 84),
                dtype=torch.uint8,
                device=device,
            ),
            "terminations": torch.zeros(
                (num_experiences,), dtype=torch.bool, device=device
            ),
            "truncations": torch.zeros(
                (num_experiences,), dtype=torch.bool, device=device
            ),
            "batch_size": [num_experiences],
        }

    @staticmethod
    def make_uncompressible_mock_data(num_experiences: int, device=None) -> dict:
        """Uncompressible data for testing."""
        if device is None:
            device = torch.device("cpu")
        return {
            "observations": torch.randn(
                (num_experiences, 4, 84, 84),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.randint(0, 10, (num_experiences,), device=device),
            "rewards": torch.randn(
                (num_experiences,), dtype=torch.float32, device=device
            ),
            "next_observations": torch.randn(
                (num_experiences, 4, 84, 84),
                dtype=torch.float32,
                device=device,
            ),
            "terminations": torch.rand((num_experiences,), device=device)
            < 0.2,  # ~20% True
            "truncations": torch.rand((num_experiences,), device=device)
            < 0.1,  # ~10% True
            "batch_size": [num_experiences],
        }

    @pytest.mark.benchmark(
        group="tensor_serialization_speed",
        min_time=0.1,
        max_time=0.5,
        min_rounds=5,
        disable_gc=True,
        warmup=False,
    )
    @pytest.mark.parametrize(
        "serialization_method",
        ["pickle", "torch.save", "untyped_storage", "numpy", "safetensors"],
    )
    def test_tensor_to_bytestream_speed(self, benchmark, serialization_method: str):
        """Benchmark the speed of different tensor serialization methods.

        TODO: we might need to also test which methods work on the gpu.
        pytest benchmarks/test_compressed_storage_benchmark.py::TestCompressedStorageBenchmark::test_tensor_to_bytestream_speed -v --benchmark-only --benchmark-sort='mean' --benchmark-columns='mean, ops'

        ------------------------ benchmark 'tensor_to_bytestream_speed': 5 tests -------------------------
        Name (time in us)                                           Mean (smaller is better)   OPS (bigger is better)
        --------------------------------------------------------------------------------------------------
        test_tensor_serialization_speed[numpy]                    2.3520 (1.0)      425,162.1779 (1.0)
        test_tensor_serialization_speed[safetensors]             14.7170 (6.26)      67,948.7129 (0.16)
        test_tensor_serialization_speed[pickle]                  19.0711 (8.11)      52,435.3333 (0.12)
        test_tensor_serialization_speed[torch.save]              32.0648 (13.63)     31,186.8261 (0.07)
        test_tensor_serialization_speed[untyped_storage]     38,227.0224 (>1000.0)       26.1595 (0.00)
        --------------------------------------------------------------------------------------------------
        """

        def serialize_with_pickle(data: torch.Tensor) -> bytes:
            """Serialize tensor using pickle."""
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            return buffer.getvalue()

        def serialize_with_untyped_storage(data: torch.Tensor) -> bytes:
            """Serialize tensor using torch's built-in method."""
            return bytes(data.untyped_storage())

        def serialize_with_numpy(data: torch.Tensor) -> bytes:
            """Serialize tensor using numpy."""
            return data.numpy().tobytes()

        def serialize_with_safetensors(data: torch.Tensor) -> bytes:
            return save({"0": data})

        def serialize_with_torch(data: torch.Tensor) -> bytes:
            """Serialize tensor using torch's built-in method."""
            buffer = io.BytesIO()
            torch.save(data, buffer)
            return buffer.getvalue()

        # Benchmark each serialization method
        if serialization_method == "pickle":
            serialize_fn = serialize_with_pickle
        elif serialization_method == "torch.save":
            serialize_fn = serialize_with_torch
        elif serialization_method == "untyped_storage":
            serialize_fn = serialize_with_untyped_storage
        elif serialization_method == "numpy":
            serialize_fn = serialize_with_numpy
        elif serialization_method == "safetensors":
            serialize_fn = serialize_with_safetensors
        else:
            raise ValueError(f"Unknown serialization method: {serialization_method}")

        data = self.make_compressible_mock_data(1).get("observations")

        # Run the actual benchmark
        benchmark(serialize_fn, data)
