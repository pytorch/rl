# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for SGLang weight synchronization schemes."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect

import pytest
import torch
import torchrl.weight_update.llm.sglang_nccl as sglang_nccl
from torch.distributed.distributed_c10d import _new_process_group_helper
from torchrl._utils import logger
from torchrl.weight_update.llm import (
    get_sglang_model_metadata,
    SGLangCollectiveTransport,
    SGLangWeightSender,
    SGLangWeightSyncScheme,
)
from torchrl.weight_update.llm.sglang_nccl import (
    _process_group_options_kwargs,
    _SGLANG_FLUSH_CACHE_ENV,
)

# Check for dependencies
_has_sglang = importlib.util.find_spec("sglang") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None


def test_process_group_options_kwargs_match_torch_signature():
    kwargs = _process_group_options_kwargs()

    assert len(kwargs) == 1
    assert next(iter(kwargs)) in inspect.signature(_new_process_group_helper).parameters
    assert next(iter(kwargs.values())) is None


@pytest.mark.gpu
@pytest.mark.skipif(not _has_sglang, reason="sglang not available")
@pytest.mark.skipif(not _has_transformers, reason="transformers not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSGLangWeightSyncScheme:
    """Tests for SGLangWeightSyncScheme configuration."""

    @pytest.fixture(scope="class")
    def model_name(self):
        """Model name for testing - small model for faster testing."""
        return "Qwen/Qwen2.5-0.5B"

    def test_scheme_initialization(self):
        """Test SGLangWeightSyncScheme initialization with valid parameters."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            num_gpus=1,
            strategy="tensordict",
            device=0,
        )

        assert scheme.server_url == "http://localhost:30000"
        assert scheme.master_address == "localhost"
        assert scheme.master_port == 29500
        assert scheme.num_gpus == 1
        assert scheme.strategy_name == "tensordict"
        assert scheme.world_size == 2  # 1 trainer + 1 gpu

    def test_scheme_auto_port(self):
        """Test that master_port is auto-assigned when not provided."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=2,
        )

        assert scheme.master_port > 0
        assert scheme.master_port <= 55535
        assert scheme.world_size == 3  # 1 trainer + 2 gpus

    def test_create_transport(self):
        """Test transport creation from scheme."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        transport = scheme.create_transport()
        assert isinstance(transport, SGLangCollectiveTransport)
        assert transport.server_url == "http://localhost:30000"
        assert transport.rank == 0
        assert transport.world_size == 2

    def test_create_sender(self):
        """Test sender creation from scheme."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        sender = scheme.create_sender()
        assert isinstance(sender, SGLangWeightSender)

    def test_create_receiver_returns_none(self):
        """Test that create_receiver returns None (SGLang manages receivers)."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        receiver = scheme.create_receiver()
        assert receiver is None


class TestSGLangFlushCacheFlag:
    """CPU-only tests for the flush_cache_on_update flag (no server needed)."""

    def _make_transport(self, **kwargs):
        return SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            **kwargs,
        )

    def test_scheme_flush_default_threads_to_transport(self):
        scheme = SGLangWeightSyncScheme(server_url="http://localhost:30000", num_gpus=1)
        assert scheme.flush_cache_on_update is None
        assert scheme.create_transport().flush_cache_on_update is None

        scheme_off = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
            flush_cache_on_update=False,
        )
        assert scheme_off.create_transport().flush_cache_on_update is False

        scheme_abort = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
            flush_cache_on_update=True,
            pause_mode="abort",
        )
        transport = scheme_abort.create_transport()
        assert transport.flush_cache_on_update is True
        assert transport.pause_mode == "abort"

    def test_flush_requires_abort_pause_mode(self):
        # SGLang only flushes when idle; retract/in_place leave requests queued.
        with pytest.raises(ValueError, match="pause_mode='abort'"):
            SGLangWeightSyncScheme(
                server_url="http://localhost:30000",
                num_gpus=1,
                flush_cache_on_update=True,
                pause_mode="retract",
            )
        with pytest.raises(ValueError, match="pause_mode='abort'"):
            self._make_transport(flush_cache_on_update=True, pause_mode="retract")
        # The default (env-resolved) pause mode is retract: explicit True
        # without an abort pause mode fails fast at transport creation.
        with pytest.raises(ValueError, match="pause_mode='abort'"):
            self._make_transport(flush_cache_on_update=True)

    def test_update_payload_flush_follows_pause_mode(self, monkeypatch):
        monkeypatch.delenv(_SGLANG_FLUSH_CACHE_ENV, raising=False)

        # Default pause mode (retract) cannot honor a flush: off.
        payload = self._make_transport()._build_update_payload(
            ["w"], ["bfloat16"], [[2]]
        )
        assert payload["flush_cache"] is False
        assert payload["abort_all_requests"] is False
        assert payload["names"] == ["w"]

        # Abort pause mode guarantees an idle scheduler: flush by default.
        payload = self._make_transport(pause_mode="abort")._build_update_payload(
            ["w"], ["bfloat16"], [[2]]
        )
        assert payload["flush_cache"] is True

        # Explicit off wins even under abort.
        payload = self._make_transport(
            flush_cache_on_update=False, pause_mode="abort"
        )._build_update_payload(["w"], ["bfloat16"], [[2]])
        assert payload["flush_cache"] is False

    def test_update_payload_env_override(self, monkeypatch):
        # The env var overrides the configured flag but cannot force a flush
        # the pause mode cannot honor.
        monkeypatch.setenv(_SGLANG_FLUSH_CACHE_ENV, "0")
        payload = self._make_transport(pause_mode="abort")._build_update_payload(
            ["w"], ["bfloat16"], [[2]]
        )
        assert payload["flush_cache"] is False

        monkeypatch.setenv(_SGLANG_FLUSH_CACHE_ENV, "1")
        payload = self._make_transport(pause_mode="abort")._build_update_payload(
            ["w"], ["bfloat16"], [[2]]
        )
        assert payload["flush_cache"] is True

        # Env-forced flush under retract is downgraded, not honored.
        payload = self._make_transport()._build_update_payload(
            ["w"], ["bfloat16"], [[2]]
        )
        assert payload["flush_cache"] is False

    def test_flush_downgrade_warning_emitted_once(self, monkeypatch):
        monkeypatch.setenv(_SGLANG_FLUSH_CACHE_ENV, "1")
        warnings_seen = []
        monkeypatch.setattr(
            sglang_nccl.torchrl_logger,
            "warning",
            lambda msg, *args, **kwargs: warnings_seen.append(msg),
        )
        transport = self._make_transport()  # default retract pause mode
        transport._build_update_payload(["w"], ["bfloat16"], [[2]])
        transport._build_update_payload(["w"], ["bfloat16"], [[2]])
        assert len(warnings_seen) == 1
        assert "Skipping the SGLang radix-cache flush" in warnings_seen[0]


@pytest.mark.gpu
@pytest.mark.skipif(not _has_sglang, reason="sglang not available")
@pytest.mark.skipif(not _has_transformers, reason="transformers not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSGLangWeightSender:
    """Tests for SGLangWeightSender."""

    @pytest.fixture(scope="class")
    def model_name(self):
        """Model name for testing - small model for faster testing."""
        return "Qwen/Qwen2.5-0.5B"

    @pytest.fixture(scope="class")
    def source_model(self, model_name):
        """Create source model for weight extraction."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )

        yield model

        # Cleanup
        try:
            del model
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_register_model(self, source_model):
        """Test model registration."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        sender = scheme.create_sender()
        sender.register_model(source_model)

        # Model reference should be stored
        assert sender._model_ref is not None
        assert sender._model_ref() is source_model

    def test_get_model_metadata(self, source_model):
        """Test model metadata extraction utility."""
        metadata = get_sglang_model_metadata(source_model)

        assert isinstance(metadata, dict)
        assert len(metadata) > 0

        # Check metadata structure
        for name, (dtype, shape) in metadata.items():
            assert isinstance(name, str)
            assert isinstance(dtype, torch.dtype)
            assert isinstance(shape, (tuple, torch.Size))

    def test_update_weights_requires_init(self, source_model):
        """Test that update_weights fails if transport not initialized."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        sender = scheme.create_sender()
        sender.register_model(source_model)

        # Should raise because transport not initialized
        with pytest.raises(RuntimeError, match="Transport not initialized"):
            sender.update_weights()

    def test_update_weights_requires_model(self):
        """Test that update_weights fails if no model registered."""
        scheme = SGLangWeightSyncScheme(
            server_url="http://localhost:30000",
            num_gpus=1,
        )

        sender = scheme.create_sender()
        # Don't register model

        # Mock the transport as initialized
        sender._transport = object()  # Fake transport

        # Should raise because no model registered
        with pytest.raises(RuntimeError, match="No model registered"):
            sender.update_weights()


@pytest.mark.gpu
@pytest.mark.skipif(not _has_sglang, reason="sglang not available")
@pytest.mark.skipif(not _has_transformers, reason="transformers not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSGLangCollectiveTransport:
    """Tests for SGLangCollectiveTransport."""

    def test_transport_initialization(self):
        """Test transport initialization with valid parameters."""
        transport = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            device=0,
        )

        assert transport.server_url == "http://localhost:30000"
        assert transport.master_address == "localhost"
        assert transport.master_port == 29500
        assert transport.rank == 0
        assert transport.world_size == 2
        assert transport.device == 0

    def test_transport_device_parsing(self):
        """Test device specification parsing."""
        # Test string device
        transport = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            device="cuda:1",
        )
        assert transport.device == 1

        # Test torch.device
        transport2 = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            device=torch.device("cuda:2"),
        )
        assert transport2.device == 2

        # Test None (defaults to 0)
        transport3 = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
            device=None,
        )
        assert transport3.device == 0

    def test_check_connection_before_init(self):
        """Test that check_connection returns False before init."""
        transport = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
        )

        assert transport.check_connection() is False

    def test_send_weights_requires_init(self):
        """Test that send_weights fails if not initialized."""
        transport = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=0,
            world_size=2,
        )

        # Should raise because comm group not initialized
        with pytest.raises(RuntimeError, match="Communication group not initialized"):
            transport.send_weights("model", {"param": torch.zeros(10)})

    def test_init_requires_rank_zero(self):
        """Test that init_all_workers_group only works for rank 0."""
        transport = SGLangCollectiveTransport(
            server_url="http://localhost:30000",
            master_address="localhost",
            master_port=29500,
            rank=1,  # Not rank 0
            world_size=2,
        )

        # Should raise because not rank 0
        with pytest.raises(RuntimeError, match="Only rank 0"):
            transport.init_all_workers_group({})


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v", "-s"] + unknown)
