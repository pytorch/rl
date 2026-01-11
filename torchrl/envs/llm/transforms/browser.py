# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Browser automation transform for LLM agents."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

from tensordict import TensorDictBase

from torchrl.envs.llm.transforms.tools import MCPToolTransform

# Schema for the browser tool
BROWSER_SCHEMA = {
    "name": "browser",
    "description": "Browse and interact with web pages",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "click",
                    "type",
                    "screenshot",
                    "extract",
                    "scroll",
                ],
                "description": "The action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for navigate action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to target element (for click/type/extract actions)",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type action)",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Amount to scroll in pixels (for scroll action)",
            },
            "extract_type": {
                "type": "string",
                "enum": ["text", "html", "attribute"],
                "description": "What to extract from the element (for extract action)",
            },
            "attribute": {
                "type": "string",
                "description": "Attribute name to extract (for extract action with extract_type=attribute)",
            },
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "navigate"}}},
                "then": {"required": ["url"]},
            },
            {
                "if": {
                    "properties": {
                        "action": {"enum": ["click", "type", "extract"]},
                    }
                },
                "then": {"required": ["selector"]},
            },
            {
                "if": {"properties": {"action": {"const": "type"}}},
                "then": {"required": ["text"]},
            },
            {
                "if": {"properties": {"action": {"const": "scroll"}}},
                "then": {"required": ["scroll_amount"]},
            },
            {
                "if": {"properties": {"action": {"const": "extract"}}},
                "then": {"required": ["extract_type"]},
            },
            {
                "if": {
                    "properties": {
                        "action": {"const": "extract"},
                        "extract_type": {"const": "attribute"},
                    }
                },
                "then": {"required": ["attribute"]},
            },
        ],
    },
}


class BrowserTransform(MCPToolTransform):
    """A transform that enables web browsing capabilities.

    This transform allows LLM agents to interact with web pages through a browser,
    supporting actions like navigation, clicking, typing, and extracting content.

    For a complete example of how to use this transform, see the :ref:`llm_tools` tutorial.

    Args:
        allowed_domains (list[str], optional): List of allowed domains. If None, all domains are allowed.
        headless (bool): Whether to run browser in headless mode. Defaults to True.
        timeout (float): Timeout for browser operations in seconds. Defaults to 30.0.
        tokenizer: The tokenizer to use. Defaults to None.
        tool_name (str): The name of the tool in chat history. Defaults to "tool".
    """

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        headless: bool = True,
        timeout: float = 30.0,
        tokenizer=None,  # type: ignore
        tool_name: str = "tool",
    ):
        self.allowed_domains = allowed_domains
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self.loop = asyncio.get_event_loop()

        super().__init__(
            tools={"browser": self._execute_browser_action},
            tool_schemas={"browser": BROWSER_SCHEMA},
            tokenizer=tokenizer,
            tool_name=tool_name,
            timeout=timeout,
        )

    async def _init_browser(self):
        """Initialize the browser if not already initialized."""
        from playwright.async_api import async_playwright

        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

    def _validate_url(self, url: str) -> bool:
        """Validate if the URL is allowed based on domain restrictions."""
        if not self.allowed_domains:
            return True

        try:
            domain = urlparse(url).netloc
            return any(domain.endswith(d) for d in self.allowed_domains)
        except Exception:
            return False

    async def _navigate(self, url: str) -> dict[str, Any]:
        """Navigate to a URL."""
        if not self._validate_url(url):
            return {
                "success": False,
                "error": f"Domain not allowed. Must be one of: {self.allowed_domains}",
            }

        try:
            await self._init_browser()
            response = await self.page.goto(url, wait_until="networkidle")
            return {
                "success": True,
                "result": {
                    "url": self.page.url,
                    "status": response.status if response else None,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _click(self, selector: str) -> dict[str, Any]:
        """Click an element on the page."""
        try:
            await self._init_browser()
            await self.page.click(selector)
            return {"success": True, "result": {"clicked": selector}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _type(self, selector: str, text: str) -> dict[str, Any]:
        """Type text into an element."""
        try:
            await self._init_browser()
            await self.page.fill(selector, text)
            return {"success": True, "result": {"typed": text, "into": selector}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _scroll(self, amount: int) -> dict[str, Any]:
        """Scroll the page."""
        try:
            await self._init_browser()
            await self.page.evaluate(f"window.scrollBy(0, {amount})")
            return {"success": True, "result": {"scrolled": amount}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract(
        self, selector: str, extract_type: str, attribute: str | None = None
    ) -> dict[str, Any]:
        """Extract content from the page."""
        try:
            await self._init_browser()
            element = await self.page.wait_for_selector(selector)
            if not element:
                return {"success": False, "error": f"Element not found: {selector}"}

            if extract_type == "text":
                content = await element.text_content()
            elif extract_type == "html":
                content = await element.inner_html()
            elif extract_type == "attribute" and attribute:
                content = await element.get_attribute(attribute)
            else:
                return {"success": False, "error": "Invalid extraction type"}

            return {
                "success": True,
                "result": {"content": content, "type": extract_type},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_browser_action_async(self, **kwargs) -> dict[str, Any]:
        """Execute a browser action asynchronously."""
        action = kwargs.pop("action")

        if action == "navigate":
            return await self._navigate(kwargs["url"])
        elif action == "click":
            return await self._click(kwargs["selector"])
        elif action == "type":
            return await self._type(kwargs["selector"], kwargs["text"])
        elif action == "scroll":
            return await self._scroll(kwargs["scroll_amount"])
        elif action == "extract":
            return await self._extract(
                kwargs["selector"],
                kwargs["extract_type"],
                kwargs.get("attribute"),
            )
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _execute_browser_action(self, **kwargs) -> dict[str, Any]:
        """Execute a browser action."""
        return self.loop.run_until_complete(
            self._execute_browser_action_async(**kwargs)
        )

    def close(self):
        """Close the browser and clean up resources."""
        if self.browser:
            self.loop.run_until_complete(self.browser.close())
            self.browser = None
            self.context = None
            self.page = None

    def __del__(self):
        """Ensure browser is closed on deletion."""
        self.close()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Reset the browser state."""
        # Close and reinitialize browser on reset
        self.close()
        return tensordict_reset

    def clone(self):
        """Clone the browser transform."""
        return self.__class__(
            allowed_domains=self.allowed_domains,
            headless=self.headless,
            timeout=self.timeout,
            tokenizer=self.tokenizer,
            tool_name=self.tool_name,
        )
