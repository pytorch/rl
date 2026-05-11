# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util

import pytest
import torch
from tensordict import set_list_to_stack, TensorDict
from torchrl.data.llm import History
from torchrl.modules.llm.policies.common import ChatHistory
from torchrl.modules.llm.policies.transformers_wrapper import TransformersWrapper

_has_transformers = importlib.import_module("transformers") is not None

# Skip all these tests if gpu is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU not available"
)


@pytest.fixture(scope="module")
def transformers_wrapper():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        model = TransformersWrapper(
            model="Qwen/Qwen2.5-0.5B",
            tokenizer="Qwen/Qwen2.5-0.5B",
            pad_model_input=False,
            generate=False,
        )
        return model


@pytest.mark.skipif(not _has_transformers, reason="transformers not installed")
class TestWrappers:
    @pytest.mark.parametrize("packing", [True, False])
    @set_list_to_stack(True)
    def test_packing(self, benchmark, transformers_wrapper, packing: bool):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.device(device):
            transformers_wrapper = TransformersWrapper(
                model=transformers_wrapper.model,
                tokenizer=transformers_wrapper.tokenizer,
                pad_model_input=not packing,
                generate=False,
                pad_output=False,
            )
            data = TensorDict(
                {
                    "history": ChatHistory(
                        full=History(
                            role=[
                                ["user", "assistant"],
                                ["user", "assistant"],
                                ["user", "assistant"],
                                ["user", "assistant"],
                            ],
                            content=[
                                [
                                    "Lorem ipsum dolor sit amet",
                                    "consectetur adipiscing elit",
                                ],
                                [
                                    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
                                    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat",
                                ],
                                [
                                    "Lorem ipsum dolor sit amet",
                                    "consectetur adipiscing elit",
                                ],
                                [
                                    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
                                    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat",
                                ],
                            ],
                            batch_size=(4, 2),
                            device=device,
                        ),
                        batch_size=(4,),
                        device=device,
                    )
                },
                batch_size=(4,),
                device=device,
            ).to_lazystack()

            def setup():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            benchmark.pedantic(
                transformers_wrapper,
                (data,),
                rounds=10,
                warmup_rounds=3,
                setup=setup,
            )


# ----- Agentic ToolCompose dispatch benchmarks -----


class _SleepyTool:
    """Bench-only tool: simulates a network/I/O call via asyncio.sleep."""

    description = "sleep"
    input_schema = {
        "type": "object",
        "properties": {"ms": {"type": "integer"}},
    }
    output_schema = None
    wants_state = False

    def __init__(self, name: str) -> None:
        self.name = name

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def run(self, args, ctx):
        import asyncio as _asyncio

        from torchrl.envs.llm.agentic import ToolResult

        await _asyncio.sleep(args.get("ms", 100) / 1000)
        return ToolResult.from_text("ok")


@pytest.mark.benchmark(group="agentic-dispatch")
@pytest.mark.parametrize("n_tools", [3, 8])
def test_toolcompose_parallel_dispatch(benchmark, n_tools):
    """Bench parallel ToolCompose dispatch.

    With ``n_tools`` async tools each sleeping 50ms, parallel dispatch
    should bottom out near 50ms regardless of ``n_tools``; serial
    dispatch would scale linearly.
    """
    from torchrl.envs import TransformedEnv
    from torchrl.envs.llm import ChatEnv
    from torchrl.envs.llm.agentic import ToolCompose
    from torchrl.envs.llm.agentic.parsers import XMLToolCallParser

    set_list_to_stack(True).set()
    base = ChatEnv(batch_size=(1,), input_mode="history")
    tools = [_SleepyTool(f"t{i}") for i in range(n_tools)]
    env = TransformedEnv(
        base, ToolCompose(tools=tools, parser=XMLToolCallParser())
    )

    fake = "".join(
        f'<tool name="t{i}" tag="{i}">{{"ms": 50}}</tool>'
        for i in range(n_tools)
    )

    def go():
        obs = env.reset(TensorDict({"query": "go"}, batch_size=(1,)))
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=fake).view(1, 1), dim=-1
        )
        env.step(obs)

    benchmark(go)


@pytest.mark.benchmark(group="agentic-dispatch")
def test_toolcompose_single_call_baseline(benchmark):
    """One-call baseline so the n=3 / n=8 numbers are interpretable."""
    from torchrl.envs import TransformedEnv
    from torchrl.envs.llm import ChatEnv
    from torchrl.envs.llm.agentic import ToolCompose
    from torchrl.envs.llm.agentic.parsers import XMLToolCallParser

    set_list_to_stack(True).set()
    base = ChatEnv(batch_size=(1,), input_mode="history")
    env = TransformedEnv(
        base,
        ToolCompose(
            tools=[_SleepyTool("t0")], parser=XMLToolCallParser()
        ),
    )
    fake = '<tool name="t0" tag="0">{"ms": 50}</tool>'

    def go():
        obs = env.reset(TensorDict({"query": "go"}, batch_size=(1,)))
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=fake).view(1, 1), dim=-1
        )
        env.step(obs)

    benchmark(go)
