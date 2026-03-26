# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for SGLang backends."""

from __future__ import annotations

import socket
import subprocess
import time
from typing import Any

import requests

from torchrl._utils import logger as torchrl_logger


def get_open_port() -> int:
    """Get an available port on localhost.

    Returns:
        int: An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(
    server_url: str,
    timeout: float = 300.0,
    check_interval: float = 1.0,
    process: subprocess.Popen | None = None,
) -> bool:
    """Wait for an SGLang server to become ready.

    Args:
        server_url: Base URL of the SGLang server (e.g., "http://localhost:30000")
        timeout: Maximum time to wait in seconds
        check_interval: Time between health checks in seconds
        process: Optional subprocess handle to check for early termination.
            If the process dies before the server becomes ready, a RuntimeError
            is raised immediately instead of waiting for the full timeout.

    Returns:
        bool: True if server is ready, False if timeout occurred

    Raises:
        TimeoutError: If the server does not become ready within the timeout
        RuntimeError: If the subprocess dies before the server becomes ready
    """
    server_url = server_url.rstrip("/")
    health_url = f"{server_url}/health"

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if subprocess died
        if process is not None:
            poll_result = process.poll()
            if poll_result is not None:
                raise RuntimeError(
                    f"SGLang server process died with exit code {poll_result}"
                )

        try:
            response = requests.get(health_url, timeout=5.0)
            if response.status_code == 200:
                torchrl_logger.info(f"SGLang server at {server_url} is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(check_interval)

    raise TimeoutError(
        f"SGLang server at {server_url} did not become ready within {timeout}s"
    )


def check_server_health(server_url: str, timeout: float = 5.0) -> bool:
    """Check if an SGLang server is healthy.

    Args:
        server_url: Base URL of the SGLang server
        timeout: Request timeout in seconds

    Returns:
        bool: True if server is healthy, False otherwise
    """
    server_url = server_url.rstrip("/")
    try:
        response = requests.get(f"{server_url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_model_info(server_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Get model information from an SGLang server.

    Args:
        server_url: Base URL of the SGLang server
        timeout: Request timeout in seconds

    Returns:
        dict: Model information including model_path, tokenizer_path, etc.

    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    server_url = server_url.rstrip("/")
    response = requests.get(f"{server_url}/model_info", timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_server_info(server_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Get server information from an SGLang server.

    Args:
        server_url: Base URL of the SGLang server
        timeout: Request timeout in seconds

    Returns:
        dict: Server information including tp_size, dp_size, etc.

    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    server_url = server_url.rstrip("/")
    response = requests.get(f"{server_url}/server_info", timeout=timeout)
    response.raise_for_status()
    return response.json()


def flush_cache(server_url: str, timeout: float = 30.0) -> bool:
    """Flush the radix cache on an SGLang server.

    This is automatically triggered when model weights are updated,
    but can be called manually if needed.

    Args:
        server_url: Base URL of the SGLang server
        timeout: Request timeout in seconds

    Returns:
        bool: True if cache was flushed successfully
    """
    server_url = server_url.rstrip("/")
    try:
        response = requests.post(f"{server_url}/flush_cache", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def convert_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 128,
    stop: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convert TorchRL sampling parameters to SGLang format.

    Args:
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling
        top_k: Top-k sampling (-1 for disabled)
        max_tokens: Maximum tokens to generate
        stop: List of stop strings
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        repetition_penalty: Repetition penalty
        **kwargs: Additional parameters passed through

    Returns:
        dict: SGLang-compatible sampling parameters
    """
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }

    if top_k > 0:
        params["top_k"] = top_k

    if stop:
        params["stop"] = stop

    # Pass through any additional parameters
    params.update(kwargs)

    return params


def dtype_to_str(dtype) -> str:
    """Convert a torch dtype to a string representation.

    Args:
        dtype: A torch.dtype object

    Returns:
        str: String representation (e.g., "float16", "bfloat16")
    """
    import torch

    dtype_map = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    return dtype_map.get(dtype, str(dtype).split(".")[-1])


def str_to_dtype(dtype_str: str):
    """Convert a string representation to a torch dtype.

    Args:
        dtype_str: String representation (e.g., "float16", "bfloat16")

    Returns:
        torch.dtype: The corresponding dtype
    """
    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    return dtype_map.get(dtype_str, torch.float32)
