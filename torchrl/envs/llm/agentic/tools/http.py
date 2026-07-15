# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""HTTP tool with built-in rate limiting.

Async-only via :mod:`urllib.request` offloaded to a worker thread (so we
don't introduce ``aiohttp``/``httpx`` as a hard dependency). For
production agentic workloads users should pair this with a
:class:`~torchrl.envs.llm.agentic.RateLimiter` keyed on the tool name in
the parent :class:`~torchrl.envs.llm.agentic.ToolCompose`.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import urlparse

from ..protocols import TextPart, ToolContext, ToolResult


_DEFAULT_MAX_BYTES = 1 << 20  # 1 MiB


class HttpTool:
    """Make an HTTP request and return the response body.

    Args:
        allowed_hosts: If non-empty, requests to hosts not in this set
            return an error result. Use ``("api.openai.com",)`` style.
            Empty disables the check (use only with a stronger
            sandbox/network policy upstream).
        timeout: Per-request timeout (seconds).
        max_response_bytes: Cap on the returned body. Larger responses
            are truncated with a marker.

    Examples:
        >>> from torchrl.envs.llm.agentic.tools.http import HttpTool
        >>> tool = HttpTool(allowed_hosts=("api.example.com",))
    """

    name: ClassVar[str] = "http"
    description: ClassVar[str] = "Make an HTTP request. Returns body, headers, status."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string"},  # default GET
            "headers": {"type": "object"},
            "body": {"type": "string"},
        },
        "required": ["url"],
    }
    output_schema: ClassVar[Mapping[str, Any] | None] = None
    wants_state: ClassVar[bool] = False

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...] = (),
        timeout: float = 10.0,
        max_response_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.allowed_hosts = tuple(allowed_hosts)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        url = args["url"]
        if not isinstance(url, str):
            return ToolResult.from_text("'url' must be a string", is_error=True)
        scheme = urlparse(url).scheme.lower()
        if scheme not in {"http", "https"}:
            return ToolResult(
                parts=(
                    TextPart(
                        text=(
                            f"URL scheme {scheme!r} is not allowed; "
                            "expected 'http' or 'https'"
                        )
                    ),
                ),
                is_error=True,
                meta={"blocked_scheme": scheme},
            )
        method = (args.get("method") or "GET").upper()
        headers = dict(args.get("headers") or {})
        body = args.get("body")
        if self.allowed_hosts:
            host = _host_of(url)
            if host not in self.allowed_hosts:
                return ToolResult(
                    parts=(
                        TextPart(
                            text=(
                                f"host {host!r} not in allowed_hosts "
                                f"{self.allowed_hosts!r}"
                            ),
                        ),
                    ),
                    is_error=True,
                    meta={"blocked_host": host},
                )
        data = body.encode("utf-8") if isinstance(body, str) else body
        try:
            status, resp_body, resp_headers, truncated = await asyncio.to_thread(
                _do_request,
                url,
                method,
                headers,
                data,
                self.timeout,
                self.max_response_bytes,
                self.allowed_hosts,
            )
        except urllib_error.HTTPError as e:
            return ToolResult(
                parts=(TextPart(text=f"HTTP {e.code}: {e.reason}"),),
                is_error=True,
                meta={"status": e.code},
            )
        except urllib_error.URLError as e:
            return ToolResult(
                parts=(TextPart(text=f"URL error: {e.reason}"),),
                is_error=True,
                meta={"error": str(e.reason)},
            )
        text = resp_body.decode("utf-8", errors="replace")
        if truncated:
            text += "\n... [truncated]"
        return ToolResult(
            parts=(TextPart(text=text),),
            is_error=status >= 400,
            meta={
                "status": status,
                "headers": dict(resp_headers),
                "truncated": truncated,
            },
        )


def _host_of(url: str) -> str:
    return urlparse(url).hostname or ""


class _SafeRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Restrict redirect schemes and optionally destination hosts."""

    def __init__(self, allowed_hosts: tuple[str, ...]) -> None:
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urlparse(newurl).scheme.lower()
        if scheme not in {"http", "https"}:
            raise urllib_error.URLError(
                f"redirect URL scheme {scheme!r} is not allowed"
            )
        host = _host_of(newurl)
        if self.allowed_hosts and host not in self.allowed_hosts:
            raise urllib_error.URLError(
                f"redirect host {host!r} not in allowed_hosts "
                f"{self.allowed_hosts!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _do_request(
    url: str,
    method: str,
    headers: Mapping[str, str],
    data: bytes | None,
    timeout: float,
    max_bytes: int,
    allowed_hosts: tuple[str, ...],
) -> tuple[int, bytes, Mapping[str, str], bool]:
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise urllib_error.URLError(f"URL scheme {scheme!r} is not allowed")
    req = urllib_request.Request(url, data=data, headers=dict(headers), method=method)
    opener = urllib_request.build_opener(_SafeRedirectHandler(allowed_hosts))
    with opener.open(req, timeout=timeout) as resp:
        body = resp.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        return resp.status, body[:max_bytes], dict(resp.headers.items()), truncated


__all__ = ["HttpTool"]
