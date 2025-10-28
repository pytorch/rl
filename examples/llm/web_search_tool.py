# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple web search tool using DuckDuckGo and SimpleToolTransform."""

import urllib.parse
import urllib.request

from tensordict import set_list_to_stack, TensorDict

from torchrl.data.llm import History
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms import SimpleToolTransform

set_list_to_stack(True).set()


def web_search(query: str) -> dict:
    """Search DuckDuckGo and return results."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8")

            results = []
            for line in html.split("\n"):
                if 'class="result__a"' in line and "href=" in line:
                    start = line.find('href="') + 6
                    end = line.find('"', start)
                    href = line[start:end]

                    title_start = line.find(">", end) + 1
                    title_end = line.find("<", title_start)
                    title = line[title_start:title_end]

                    if href and title and len(results) < 5:
                        results.append({"title": title, "url": href})

            return {"success": True, "query": query, "results": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_webpage(url: str) -> dict:
    """Fetch webpage content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8")

            title = ""
            if "<title>" in html:
                start = html.find("<title>") + 7
                end = html.find("</title>", start)
                title = html[start:end]

            text = html
            for tag in ["<script", "<style", "<nav", "<footer"]:
                while tag in text:
                    start = text.find(tag)
                    end = text.find(">", start) + 1
                    close_tag = f"</{tag[1:]}"
                    if close_tag in text[end:]:
                        end = text.find(close_tag, end) + len(close_tag) + 1
                    text = text[:start] + text[end:]

            for tag in ["<", ">"]:
                text = text.replace(tag, " ")

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            content = " ".join(lines)[:3000]

            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content,
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    tools = {"search": web_search, "fetch": fetch_webpage}

    env = ChatEnv(batch_size=(1,))
    env = env.append_transform(SimpleToolTransform(tools=tools))

    reset_data = TensorDict(query="You are a helpful assistant", batch_size=(1,))
    td = env.reset(reset_data)

    history = td.get("history")

    assistant_response = (
        History(
            role="assistant",
            content='Let me search for PyTorch tutorials.\n<tool>search\n{"query": "pytorch tutorial"}</tool>',
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    history.full = history.prompt.extend(assistant_response, inplace=True, dim=-1)
    history.response = assistant_response

    result = env.step(td.set("history", history))

    print("Search executed successfully!")
    print("\nTool response:")
    tool_response = result["next", "history"].prompt[-1]
    print(f"Role: {tool_response.role}")
    print(f"Content: {tool_response.content[:500]}...")

    fetch_response = (
        History(
            role="assistant",
            content='<tool>fetch\n{"url": "https://pytorch.org"}</tool>',
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    history = result["next", "history"]
    history.full = history.prompt.extend(fetch_response, inplace=True, dim=-1)
    history.response = fetch_response

    result2 = env.step(result["next"].set("history", history))

    print("\n\nFetch executed successfully!")
    print("\nTool response:")
    fetch_tool_response = result2["next", "history"].prompt[-1]
    print(f"Role: {fetch_tool_response.role}")
    print(f"Content: {fetch_tool_response.content[:500]}...")
