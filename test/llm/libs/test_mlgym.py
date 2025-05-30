# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

from functools import partial

import pytest

from torchrl import logger as torchrl_logger
from torchrl.envs import SerialEnv

from torchrl.envs.llm import make_mlgym
from torchrl.modules.llm import TransformersWrapper


class TestMLGYM:
    def test_mlgym_specs(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.eos_token = "<|im_end|>"
        policy = TransformersWrapper(
            AutoModelForCausalLM.from_pretrained(model_name).cuda(),
            tokenizer=tokenizer,
            from_text=True,
            generate=True,
            device="cuda:0",
            generate_kwargs={
                # "temperature": 0.8,
                # "repetition_penalty": 1.5,
                "max_new_tokens": 1024
            },
        )

        env = SerialEnv(
            1,
            [
                partial(
                    make_mlgym,
                    task="prisonersDilemma",
                    tokenizer=tokenizer,
                    device="cuda:0",
                )
            ],
        )
        rollout = env.rollout(3, policy)
        torchrl_logger.info(f"{rollout=}")
        env.check_env_specs(break_when_any_done="both")

    def test_mlgym_task_reset(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.eos_token = "<|im_end|>"
        policy = TransformersWrapper(
            AutoModelForCausalLM.from_pretrained(model_name).cuda(),
            tokenizer=tokenizer,
            from_text=True,
            generate=True,
            device="cuda:0",
            generate_kwargs={
                # "temperature": 0.8,
                # "repetition_penalty": 1.5,
                "max_new_tokens": 1024
            },
        )

        env = SerialEnv(
            1,
            [
                partial(
                    make_mlgym,
                    tasks=[
                        "prisonersDilemma",
                        "regressionKaggleHousePrice",
                        "battleOfSexes",
                    ],
                    tokenizer=tokenizer,
                    device="cuda:0",
                )
            ],
        )
        # We should get at least two tasks
        rollout = env.rollout(100, policy, break_when_any_done=False)
        torchrl_logger.info(f"{rollout=}")
        torchrl_logger.info(rollout["task"])

    def test_mlgym_wrong_format(self):
        # A vanilla policy will not output anything useful, yet the env should run without error
        ...


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
