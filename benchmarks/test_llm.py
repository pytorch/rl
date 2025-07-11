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
