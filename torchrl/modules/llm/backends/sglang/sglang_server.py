# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang server-based inference service for TorchRL.

This module provides AsyncSGLang, a server-based SGLang backend that
communicates with SGLang servers via HTTP for generation and uses
NCCL for weight synchronization in RL training workflows.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import requests
import torch

from torchrl._utils import logger as torchrl_logger

from .base import RLSGLangEngine
from .sglang_utils import (
    check_server_health,
    convert_sampling_params,
    dtype_to_str,
    get_model_info,
    get_open_port,
    get_server_info,
    wait_for_server,
)


class AsyncSGLang(RLSGLangEngine):
    """Server-based SGLang inference service for TorchRL.

    AsyncSGLang provides a unified interface for text generation using SGLang servers,
    supporting both managed (subprocess) and external server modes. It integrates
    seamlessly with TorchRL's RL training workflows through NCCL-based weight
    synchronization.

    Key Features:
        - HTTP-based generation via SGLang's native /generate API
        - Cache-aware load balancing through SGLang Router
        - NCCL-based weight synchronization for RL training
        - Support for both managed and external server modes
        - Compatible interface with vLLM backends for easy migration

    Args:
        server_url: URL of an external SGLang server (e.g., "http://localhost:30000").
            If None, a managed server will be launched.
        model_path: Path or name of the model to load (for managed mode).
        tp_size: Tensor parallel size (default: 1).
        dp_size: Data parallel size (default: 1).
        timeout: Request timeout in seconds (default: 300).
        **server_kwargs: Additional arguments passed to SGLang server launch.

    Examples:
        >>> # Connect to an existing SGLang server
        >>> service = AsyncSGLang.connect("http://localhost:30000")
        >>> result = service.generate("Hello, world!")
        >>>
        >>> # Launch a managed SGLang server
        >>> service = AsyncSGLang.from_pretrained("Qwen/Qwen2.5-3B")
        >>> result = service.generate("Hello, world!")
        >>>
        >>> # With custom parameters
        >>> service = AsyncSGLang.from_pretrained(
        ...     "Qwen/Qwen2.5-7B",
        ...     tp_size=2,
        ...     max_model_len=4096
        ... )

    Note:
        For RL training with weight updates, use the weight synchronization
        methods after initializing the NCCL communication group.
    """

    def __init__(
        self,
        server_url: str | None = None,
        model_path: str | None = None,
        tp_size: int = 1,
        dp_size: int = 1,
        timeout: float = 300.0,
        **server_kwargs: Any,
    ):
        self._server_url = server_url
        self._model_path = model_path
        self._tp_size = tp_size
        self._dp_size = dp_size
        self._timeout = timeout
        self._server_kwargs = server_kwargs

        self._managed_process: subprocess.Popen | None = None
        self._log_file_path: str | None = None
        self._server_info: dict[str, Any] | None = None
        self._model_info: dict[str, Any] | None = None

        # Weight sync state
        self._master_address: str = "localhost"
        self._master_port: int | None = None
        self._weight_update_initialized: bool = False

        # If server_url is provided, we're in external mode
        if self._server_url is not None:
            self._validate_and_connect()

    def _validate_and_connect(self) -> None:
        """Validate connection to an external SGLang server."""
        if not check_server_health(self._server_url, timeout=10.0):
            raise ConnectionError(
                f"Cannot connect to SGLang server at {self._server_url}. "
                "Ensure the server is running and accessible."
            )

        # Fetch server and model info
        self._server_info = get_server_info(self._server_url)
        self._model_info = get_model_info(self._server_url)

        # Extract parallelism info from server
        self._tp_size = self._server_info.get("tp_size", 1)
        self._dp_size = self._server_info.get("dp_size", 1)
        self._model_path = self._model_info.get("model_path", self._model_path)

        torchrl_logger.info(
            f"Connected to SGLang server at {self._server_url} "
            f"(model={self._model_path}, tp={self._tp_size}, dp={self._dp_size})"
        )

    def _launch_server(self) -> None:
        """Launch a managed SGLang server subprocess."""
        if self._model_path is None:
            raise ValueError("model_path is required to launch a managed SGLang server")

        # Find an available port
        port = self._server_kwargs.pop("port", None) or get_open_port()
        host = self._server_kwargs.pop("host", "127.0.0.1")

        # Build command line arguments
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--tp-size",
            str(self._tp_size),
            "--dp-size",
            str(self._dp_size),
        ]

        # Add additional kwargs
        for key, value in self._server_kwargs.items():
            # Convert underscores to hyphens for CLI args
            cli_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
            else:
                cmd.extend([cli_key, str(value)])

        torchrl_logger.info(f"Launching SGLang server: {' '.join(cmd)}")

        # Create a temporary file for server output logging
        # This provides better visibility than PIPE, especially in CI environments
        log_file = tempfile.NamedTemporaryFile(
            mode="w", suffix="_sglang.log", delete=False
        )
        self._log_file_path = log_file.name
        torchrl_logger.info(f"SGLang server output logging to: {self._log_file_path}")

        # Launch subprocess with output going to the log file
        self._managed_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        torchrl_logger.info(
            f"SGLang server launched with PID {self._managed_process.pid}"
        )

        self._server_url = f"http://{host}:{port}"

        # Register cleanup handler
        atexit.register(self.shutdown)

        # Early crash detection: check if process is still alive after 2 seconds
        time.sleep(2)
        poll_result = self._managed_process.poll()
        if poll_result is not None:
            # Process crashed immediately - read log file and raise with output
            with open(self._log_file_path) as f:
                output = f.read()
            raise RuntimeError(
                f"SGLang server crashed immediately (exit={poll_result}):\n{output}"
            )

        # Wait for server to be ready, passing process handle for liveness checks
        try:
            wait_for_server(
                self._server_url, timeout=self._timeout, process=self._managed_process
            )
        except (TimeoutError, RuntimeError):
            # Capture server output for debugging
            if self._log_file_path and os.path.exists(self._log_file_path):
                with open(self._log_file_path) as f:
                    output = f.read()
                # Log last 20KB of output
                torchrl_logger.error(
                    f"SGLang server output (last 20000 chars):\n{output[-20000:]}"
                )

            if self._managed_process is not None:
                poll_result = self._managed_process.poll()
                if poll_result is not None:
                    torchrl_logger.error(
                        f"SGLang server process exited with code {poll_result}"
                    )
                else:
                    torchrl_logger.error(
                        "SGLang server process is still running but not responding"
                    )

            self.shutdown()
            raise

        # Fetch server info
        self._validate_and_connect()

        torchrl_logger.info(f"Managed SGLang server launched at {self._server_url}")

    @classmethod
    def connect(cls, server_url: str) -> AsyncSGLang:
        """Connect to an existing SGLang server.

        Args:
            server_url: URL of the SGLang server (e.g., "http://localhost:30000")

        Returns:
            AsyncSGLang: Connected service instance

        Raises:
            ConnectionError: If the server is not reachable
        """
        return cls(server_url=server_url)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tp_size: int = 1,
        dp_size: int = 1,
        **kwargs: Any,
    ) -> AsyncSGLang:
        """Create an AsyncSGLang instance by launching a managed server.

        Args:
            model_name: Model name or path to load
            tp_size: Tensor parallel size
            dp_size: Data parallel size
            **kwargs: Additional server arguments

        Returns:
            AsyncSGLang: Service with managed server

        Example:
            >>> service = AsyncSGLang.from_pretrained(
            ...     "Qwen/Qwen2.5-3B",
            ...     tp_size=2,
            ...     max_model_len=4096
            ... )
        """
        instance = cls(
            model_path=model_name,
            tp_size=tp_size,
            dp_size=dp_size,
            **kwargs,
        )
        instance._launch_server()
        return instance

    @property
    def server_url(self) -> str:
        """Get the server URL."""
        if self._server_url is None:
            raise RuntimeError(
                "Server URL not set. Call connect() or from_pretrained()."
            )
        return self._server_url

    def generate(
        self,
        prompts: str | list[str] | None = None,
        sampling_params: dict[str, Any] | None = None,
        *,
        input_ids: list[int] | list[list[int]] | None = None,
        return_logprobs: bool = False,
        return_text: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Generate text completions from text prompts or token IDs.

        You can provide either `prompts` (text) OR `input_ids` (tokens), but not both.

        Args:
            prompts: Input text prompt(s) for generation. Mutually exclusive with input_ids.
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, etc.)
            input_ids: Input token ID(s) for generation. Can be a single list of ints
                or a list of lists for batch generation. Mutually exclusive with prompts.
            return_logprobs: Whether to return log probabilities
            return_text: Whether to return generated text
            timeout: Request timeout in seconds
            **kwargs: Additional sampling parameters (temperature, max_new_tokens, etc.)
                These are merged into sampling_params for convenience.

        Returns:
            dict or list[dict]: Generation results with 'text', 'output_ids', 'meta_info'

        Example:
            >>> # Generate from text
            >>> result = service.generate(
            ...     "What is the capital of France?",
            ...     {"temperature": 0.7, "max_tokens": 100}
            ... )
            >>> print(result["text"])

            >>> # Generate from token IDs
            >>> result = service.generate(
            ...     input_ids=[1, 2, 3, 4],
            ...     sampling_params={"max_tokens": 50}
            ... )
            >>> print(result["output_ids"])

            >>> # Using kwargs for sampling params
            >>> result = service.generate("Hello", max_new_tokens=50, temperature=0.7)
        """
        # Validate inputs: must provide exactly one of prompts or input_ids
        if prompts is None and input_ids is None:
            raise ValueError("Must provide either 'prompts' or 'input_ids'")
        if prompts is not None and input_ids is not None:
            raise ValueError("Cannot provide both 'prompts' and 'input_ids'")

        if sampling_params is None:
            sampling_params = {}

        # Merge kwargs into sampling_params for convenience
        # Handle common aliases like max_new_tokens -> max_tokens
        merged_params = {**sampling_params, **kwargs}
        if "max_new_tokens" in merged_params and "max_tokens" not in merged_params:
            merged_params["max_tokens"] = merged_params.pop("max_new_tokens")

        # Convert to SGLang format
        sglang_params = convert_sampling_params(**merged_params)
        if return_logprobs:
            sglang_params["return_logprob"] = True

        timeout = timeout or self._timeout

        # Determine if using text or token mode
        use_tokens = input_ids is not None

        if use_tokens:
            # Handle single sequence vs batch for input_ids
            single_input = isinstance(input_ids[0], int) if input_ids else False
            if single_input:
                inputs = [input_ids]
            else:
                inputs = input_ids
        else:
            # Handle single prompt vs batch for text
            single_input = isinstance(prompts, str)
            if single_input:
                inputs = [prompts]
            else:
                inputs = prompts

        results = []
        for inp in inputs:
            if use_tokens:
                data = {
                    "input_ids": inp,
                    **sglang_params,
                }
            else:
                data = {
                    "text": inp,
                    **sglang_params,
                }

            response = requests.post(
                f"{self.server_url}/generate",
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
            results.append(response.json())

        return results[0] if single_input else results

    def generate_batch(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate text completions for a batch of prompts.

        This is an alias for generate() with a list of prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            **kwargs: Additional arguments passed to generate()

        Returns:
            list[dict]: List of generation results
        """
        return self.generate(prompts, sampling_params, **kwargs)

    def flush_cache(self) -> bool:
        """Flush the radix cache on the server.

        This is automatically triggered when weights are updated.

        Returns:
            bool: True if cache was flushed successfully
        """
        try:
            response = requests.post(
                f"{self.server_url}/flush_cache",
                timeout=30.0,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # RLSGLangEngine interface implementation

    def get_tp_size(self) -> int:
        """Get the tensor parallel size."""
        return self._tp_size

    def get_dp_size(self) -> int:
        """Get the data parallel size."""
        return self._dp_size

    def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model parameter metadata.

        Note: This requires fetching from the server. For now, returns empty dict
        and expects metadata to be provided externally.
        """
        # TODO: Implement metadata fetching from SGLang server if supported
        torchrl_logger.warning(
            "AsyncSGLang.get_model_metadata() not yet implemented - returning empty dict. "
            "Provide metadata externally during weight updates."
        )
        return {}

    def get_master_address(self) -> str:
        """Get the master address for weight synchronization."""
        return self._master_address

    def get_master_port(self) -> int:
        """Get the master port for weight synchronization."""
        if self._master_port is None:
            self._master_port = get_open_port()
        return self._master_port

    def init_weight_update_group(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        """Initialize the NCCL weight update group via SGLang's HTTP API.

        This calls the SGLang server's /init_weights_update_group endpoint
        to set up NCCL communication for weight synchronization.

        Args:
            master_address: Master address for NCCL (default: "localhost")
            master_port: Master port for NCCL (auto-assigned if None)
        """
        if master_address is not None:
            self._master_address = master_address
        if master_port is not None:
            self._master_port = master_port
        else:
            self._master_port = self.get_master_port()

        torchrl_logger.info(
            f"Initializing SGLang weight update group: "
            f"address={self._master_address}, port={self._master_port}"
        )

        # Call SGLang's init_weights_update_group API
        data = {
            "master_address": self._master_address,
            "master_port": self._master_port,
            "rank_offset": 1,  # Workers start from rank 1
            "world_size": 1 + self._tp_size * self._dp_size,  # Trainer + workers
        }

        response = requests.post(
            f"{self.server_url}/init_weights_update_group",
            json=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                f"Failed to initialize weight update group: {result.get('message', 'Unknown error')}"
            )

        self._weight_update_initialized = True
        torchrl_logger.info("SGLang weight update group initialized successfully")

    def update_weights_from_distributed(
        self,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> None:
        """Signal the server to receive a weight update via NCCL broadcast.

        This calls SGLang's /update_weights_from_distributed endpoint to
        coordinate weight reception.

        Args:
            name: Name of the parameter to update
            dtype: Data type of the tensor
            shape: Shape of the tensor
        """
        if not self._weight_update_initialized:
            raise RuntimeError(
                "Weight update group not initialized. Call init_weight_update_group() first."
            )

        data = {
            "name": name,
            "dtype": dtype_to_str(dtype),
            "shape": list(shape),
        }

        response = requests.post(
            f"{self.server_url}/update_weights_from_distributed",
            json=data,
            timeout=self._timeout,
        )
        response.raise_for_status()

    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights via NCCL broadcast.

        This method coordinates with the SGLang server to broadcast weights
        from the trainer (rank 0) to all workers.

        Args:
            weights: Iterator yielding (parameter_name, tensor) tuples
        """
        if not self._weight_update_initialized:
            raise RuntimeError(
                "Weight update group not initialized. Call init_weight_update_group() first."
            )

        # Convert iterator to dict
        weights_dict = dict(weights)

        if not weights_dict:
            torchrl_logger.warning("No weights provided for update")
            return

        torchrl_logger.info(
            f"Updating {len(weights_dict)} parameters via NCCL broadcast"
        )

        # Get NCCL communicator (must be set up by the caller)
        if not hasattr(self, "_nccl_group") or self._nccl_group is None:
            raise RuntimeError(
                "NCCL group not set up. Use SGLangWeightSyncScheme for weight synchronization."
            )

        t0 = time.time()

        for name, weight in weights_dict.items():
            # Ensure weight is on GPU
            if weight.device.type != "cuda":
                weight = weight.to("cuda:0", non_blocking=True)

            # Step 1: Signal server to expect this weight
            self.update_weights_from_distributed(
                name, weight.dtype, tuple(weight.shape)
            )

            # Step 2: Broadcast the weight via NCCL
            self._nccl_group.broadcast(
                weight, src=0, stream=torch.cuda.current_stream()
            )

        torch.cuda.synchronize()

        # Flush cache after weight update
        self.flush_cache()

        t1 = time.time()
        torchrl_logger.info(f"Updated {len(weights_dict)} weights in {t1 - t0:.3f}s")

    def shutdown(self) -> None:
        """Shutdown the managed SGLang server if running."""
        if self._managed_process is not None:
            torchrl_logger.info("Shutting down managed SGLang server...")
            self._managed_process.terminate()
            try:
                self._managed_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._managed_process.kill()
            self._managed_process = None
            torchrl_logger.info("SGLang server shutdown complete")

        # Clean up the log file
        if self._log_file_path is not None and os.path.exists(self._log_file_path):
            try:
                os.unlink(self._log_file_path)
            except OSError:
                pass  # Ignore cleanup errors
            self._log_file_path = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        managed = "managed" if self._managed_process is not None else "external"
        return (
            f"AsyncSGLang("
            f"url={self._server_url!r}, "
            f"model={self._model_path!r}, "
            f"tp={self._tp_size}, "
            f"dp={self._dp_size}, "
            f"mode={managed})"
        )
