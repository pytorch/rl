# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pytest
import torch
from mocking_classes import DummyStrDataLoader

from tensordict import lazy_stack, set_capture_non_tensor_stack, TensorDict
from torchrl.data import LazyStackStorage, ReplayBuffer, Unbounded
from torchrl.envs import Transform
from torchrl.envs.llm import LLMEnv
from torchrl.modules.llm import TransformersWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm.grpo import GRPOLoss, GRPOLossOutput, MCAdvantage

_has_transformers = importlib.util.find_spec("transformers") is not None
prompts = [
    "Lorem ipsum dolor sit amet,",
    "consectetur adipiscing elit,",
    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation",
    "ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore",
    "eu fugiat nulla pariatur.",
]


@pytest.mark.parametrize("ndim", [1, 2])
def test_mc_advantage(ndim):
    # make trajectories
    def make_silly_trajectory(n_steps=None):
        while True:
            if n_steps is None:
                n_steps = torch.randint(low=2, high=100, size=(1,)).item()
            tds = []
            for _ in range(n_steps):
                n_tokens = torch.randint(low=1, high=100, size=(1,)).item()
                rewards = [torch.randn(n_tokens, 1)]
                prompt = np.random.choice(prompts)
                td = TensorDict(
                    text=prompt,
                    next=TensorDict(
                        reward=rewards, done=torch.zeros(1, dtype=torch.bool)
                    ),
                )
                tds.append(td)
            tds[-1]["next", "done"] = torch.ones(1, dtype=torch.bool)
            yield lazy_stack(tds)

    rb = ReplayBuffer(storage=LazyStackStorage(100))
    rb.append_transform(MCAdvantage(grpo_size=4))
    if ndim == 1:
        gen = make_silly_trajectory()
        for _ in range(100):
            trajectory = next(gen)
            rb.extend(trajectory)
        assert len(rb)
        s = rb.sample(1)
        assert "advantage" in s.keys()
    else:
        gen = make_silly_trajectory(n_steps=5)
        for _ in range(100):
            trajectory = lazy_stack([next(gen) for _ in range(3)])
            trajectory = trajectory.view(-1)
            rb.extend(trajectory)
        assert len(rb)
        s = rb.sample(1)
        assert "advantage" in s.keys()


def test_grpo():
    ...


class TestPPO4LLMs:
    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test PPO with LLMs"
    )
    @set_capture_non_tensor_stack(False)
    @pytest.mark.parametrize("from_text", [True, False])
    @pytest.mark.parametrize("cls", [ClipPPOLoss, GRPOLoss])
    def test_hf(self, from_text, cls):
        from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token

        model = OPTForCausalLM(OPTConfig()).eval()
        policy_inference = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=True,
            from_text=from_text,
            return_log_probs=True,
        )
        policy_train = TransformersWrapper(
            model, tokenizer=tokenizer, generate=False, from_text=False
        )
        for p in policy_train.parameters():
            assert p.requires_grad
        # Create some fake data
        dl = DummyStrDataLoader(batch_size=32)
        llm_env = LLMEnv.from_dataloader(
            dl,
            tokenizer=tokenizer if not from_text else None,
            batch_size=(32,),
            from_text=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        class RewardTransform(Transform):
            def _step(self, td, next_td):
                next_td["reward"] = torch.randn_like(
                    td["tokens_response"], dtype=torch.float
                ).unsqueeze(-1)
                return next_td

            def transform_reward_spec(self, reward_spec):
                return reward_spec.set(
                    "reward", Unbounded((*reward_spec.shape, -1, 1), dtype=torch.float)
                )

        llm_env = llm_env.append_transform(RewardTransform())
        with torch.no_grad():
            data = llm_env.rollout(3, policy_inference)
            data = data.view(-1)
            assert data["tokens_response"].shape[-1] == 20
        # Make some fake advantages:
        data["advantage"] = torch.randn_like(data["next", "reward"])

        loss = cls(
            actor_network=policy_train,
        )
        loss_vals = loss(data)
        if cls is ClipPPOLoss:
            assert "loss_objective" in loss_vals
            assert "loss_entropy" in loss_vals
            assert loss_vals["loss_objective"].requires_grad
            assert loss_vals["loss_entropy"].requires_grad
            assert "clip_fraction" in loss_vals
            assert "kl_approx" in loss_vals
            assert "entropy" in loss_vals
            assert "ESS" in loss_vals
            assert "loss_critic" not in loss_vals
        else:
            assert isinstance(loss_vals, GRPOLossOutput)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
