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
from mocking_classes_llm import DummyStrDataLoader
from tensordict import (
    lazy_stack,
    NonTensorStack,
    set_capture_non_tensor_stack,
    TensorDict,
)
from torchrl.data import History, LazyStackStorage, ReplayBuffer, Unbounded
from torchrl.envs import Transform
from torchrl.envs.llm import LLMEnv
from torchrl.envs.llm.transforms.kl import RetrieveLogProb
from torchrl.modules.llm import TransformersWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm.dpo import DPOLoss, DPOLossOutput
from torchrl.objectives.llm.grpo import GRPOLoss, GRPOLossOutput, MCAdvantage
from torchrl.objectives.llm.sft import SFTLoss

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


@pytest.fixture(autouse=True, scope="module")
def set_list_to_stack():
    import tensordict

    with tensordict.set_list_to_stack(True):
        yield


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


class TestSFT:
    @pytest.fixture(scope="class")
    def data(self):
        from transformers import AutoTokenizer

        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you very much!",
                },
            ],
        ]
        history = History.from_chats(chats)
        assert history.shape == (2, 3)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token
        text = history[:, :-1].apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        )
        text_response = history.apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        )
        text_response = [
            txt[len(txt_start) :] for txt, txt_start in zip(text_response, text)
        ]
        td = TensorDict(
            text=text,
            text_response=text_response,
            history=history,
            next=TensorDict(
                reward=torch.randn(2, 1),
                done=torch.zeros(2, dtype=torch.bool),
                history=history,
            ),
            batch_size=(2,),
        )
        yield lazy_stack(list(td.unbind(0)))

    def tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.chat_template = _CHAT_TEMPLATES["qwen"]
        return tokenizer

    @pytest.fixture(scope="class")
    def policy_train(self):
        from transformers import OPTConfig, OPTForCausalLM

        tokenizer = self.tokenizer()
        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            chat_template_name="qwen",
        )

        return policy_train, tokenizer

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test SFT"
    )
    @pytest.mark.parametrize("loss_function", ["sft", "minor_sft"])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("normalize_by_seq_length", [True, False])
    @pytest.mark.parametrize("kl_to_ref_coeff", [None, 0.1])
    def test_sft(
        self,
        loss_function,
        reduction,
        normalize_by_seq_length,
        kl_to_ref_coeff,
        data,
        policy_train,
    ):
        policy_train, tokenizer = policy_train
        loss = SFTLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            reduction=reduction,
            normalize_by_seq_length=normalize_by_seq_length,
            kl_to_ref_coeff=kl_to_ref_coeff if loss_function != "minor_sft" else None,
            loss_function=loss_function,
            beta=0.1,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )

        td = data
        if kl_to_ref_coeff is not None or loss_function == "minor_sft":
            policy_ref = TransformersWrapper(
                policy_train.model,
                tokenizer=tokenizer,
                generate=False,
                from_text=True,
                return_log_probs=True,
                chat_template_name="qwen",
            )
            transform = RetrieveLogProb(
                policy_ref,
                assistant_only=True,
                tokenizer_kwargs={"chat_template_name": "qwen"},
                tokenizer=tokenizer,
            )
            with torch.no_grad():
                # Compute ref log-probs
                transform(td)

        loss_vals = loss(td)
        if kl_to_ref_coeff is not None and loss_function != "minor_sft":
            assert loss_vals.loss_kl_to_ref.shape == ()
            assert loss_vals.kl_to_ref.shape == ()
        if reduction == "mean":
            assert loss_vals.loss_sft.shape == ()
        elif reduction == "sum":
            assert loss_vals.loss_sft.shape == ()
        elif reduction == "none":
            assert loss_vals.loss_sft.shape == (2,)
        assert loss_vals.sum(reduce=True).shape == ()

    def test_sft_assistant_only(self, data):
        from torchrl.data.llm.chat import _CHAT_TEMPLATES
        from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]

        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            chat_template_name="qwen",
        )
        policy_ref = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            return_log_probs=True,
            chat_template_name="qwen",
        )
        transform = RetrieveLogProb(
            policy_ref,
            assistant_only=True,
            tokenizer_kwargs={"chat_template_name": "qwen"},
            tokenizer=tokenizer,
        )
        td = transform(data)
        assert td is data
        loss = SFTLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            reduction="mean",
            normalize_by_seq_length=True,
            kl_to_ref_coeff=0.1,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )
        loss(td)


class TestDPO:
    @pytest.fixture(scope="class")
    def preference_data(self):
        from transformers import AutoTokenizer

        # Create preference data with chosen/rejected pairs
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},  # chosen
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "I don't know."},  # rejected
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum physics."},
                {
                    "role": "assistant",
                    "content": "Quantum physics is complex.",
                },  # chosen
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum physics."},
                {"role": "assistant", "content": "2+2 equals 4."},  # chosen
            ],
        ]
        # with LLMs, rewards have 2 singleton dimensions
        rewards = torch.tensor([1.0, -1.0, 1.0, -1.0]).unsqueeze(-1)
        history = History.from_chats(chats)
        assert history.shape == (4, 3)  # 2 conversations, 4 messages each

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token

        # Create preference labels (True for chosen, False for rejected)
        is_chosen = torch.tensor([True, False, True, False])

        # Prepare text for each response
        text = history[:, :-2].apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        )
        text_chosen = history[:, -2:-1].apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        )
        text_rejected = history[:, -1:].apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        )

        # Create tensordict with preference data
        # We have 4 trajectories of 1 step each
        td = TensorDict(
            history=history,
            done=torch.zeros(4, dtype=torch.bool),
            next=TensorDict(
                is_chosen=is_chosen,
                done=torch.ones(4, dtype=torch.bool),
                reward=rewards,
                history=history,
            ),
            batch_size=(4,),
        ).unsqueeze(
            1
        )  # unsqueeze time dim - there's a single step
        yield lazy_stack(list(td.unbind(0)))

    @pytest.fixture(scope="class")
    def policy_train(self):
        from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token

        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            chat_template_name="qwen",
        )

        return policy_train, tokenizer

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test DPO"
    )
    @pytest.mark.parametrize("beta", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("normalize_by_seq_length", [True, False])
    @pytest.mark.parametrize("kl_to_ref_coeff", [None, 0.1])
    def test_dpo(
        self,
        beta,
        reduction,
        normalize_by_seq_length,
        kl_to_ref_coeff,
        preference_data,
        policy_train,
    ):
        policy_train, tokenizer = policy_train

        loss = DPOLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            beta=beta,
            reduction=reduction,
            normalize_by_seq_length=normalize_by_seq_length,
            kl_to_ref_coeff=kl_to_ref_coeff,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )

        td = preference_data

        # Add reference log probabilities if needed
        if kl_to_ref_coeff is not None:
            policy_ref = TransformersWrapper(
                policy_train.model,
                tokenizer=tokenizer,
                generate=False,
                from_text=True,
                return_log_probs=True,
                chat_template_name="qwen",
            )
            transform = RetrieveLogProb(
                policy_ref,
                assistant_only=True,
                tokenizer_kwargs={"chat_template_name": "qwen"},
                tokenizer=tokenizer,
            )
            with torch.no_grad():
                # Compute ref log-probs
                transform(td)

        loss_vals = loss(td)

        # Check output structure
        assert isinstance(loss_vals, DPOLossOutput)
        assert loss_vals.loss_dpo.requires_grad
        assert loss_vals.chosen_rewards is not None
        assert loss_vals.rejected_rewards is not None
        assert loss_vals.accuracy is not None

        # Check shapes based on reduction
        if reduction == "mean":
            assert loss_vals.loss_dpo.shape == ()
        elif reduction == "sum":
            assert loss_vals.loss_dpo.shape == ()
        elif reduction == "none":
            # Should have shape matching the number of preference pairs
            assert loss_vals.loss_dpo.shape == (2,)

        # Check KL loss if enabled
        if kl_to_ref_coeff is not None:
            assert loss_vals.loss_kl_to_ref is not None
            assert loss_vals.kl_to_ref is not None
            assert loss_vals.loss_kl_to_ref.shape == ()
            assert loss_vals.kl_to_ref.shape == ()
        else:
            assert loss_vals.loss_kl_to_ref is None
            assert loss_vals.kl_to_ref is None

        # Check that total loss can be computed
        total_loss = loss_vals.sum(reduce=True)
        assert total_loss.shape == ()
        assert total_loss.requires_grad

        # Check accuracy is reasonable (should be between 0 and 1)
        assert 0.0 <= loss_vals.accuracy.item() <= 1.0

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test DPO"
    )
    def test_dpo_no_preference_pairs(self, policy_train):
        """Test that DPO raises an error when no preference pairs are present."""
        policy_train, tokenizer = policy_train

        # Create data with only chosen responses (no rejected)
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello?"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ]
        history = History.from_chats(chats)

        # All responses marked as chosen (no rejected)
        is_chosen = torch.tensor([True])

        td = TensorDict(
            history=history,
            next=TensorDict(
                is_chosen=is_chosen,
                done=torch.zeros(1, dtype=torch.bool),
                history=history,
            ),
            batch_size=(1,),
        )

        loss = DPOLoss(
            actor_network=policy_train,
            tokenizer=tokenizer,
            beta=0.1,
            tokenizer_kwargs={"chat_template_name": "qwen"},
        )

        with pytest.raises(
            ValueError, match="Both chosen and rejected responses must be present"
        ):
            loss(td)

    def test_dpo_loss_function(self, preference_data):
        """Test the standalone dpo_loss function."""
        from torchrl.objectives.llm.dpo import dpo_loss

        # Create some dummy log probabilities
        policy_chosen_logprob = torch.tensor([1.0, 2.0]).requires_grad_(True)
        policy_rejected_logprob = torch.tensor([0.5, 1.0]).requires_grad_(True)
        reference_chosen_logprob = torch.tensor([0.8, 1.5]).requires_grad_(False)
        reference_rejected_logprob = torch.tensor([0.3, 0.8]).requires_grad_(False)
        beta = 0.1

        # Test different reductions
        for reduction in ["mean", "sum", "none"]:
            loss = dpo_loss(
                policy_chosen_logprob,
                policy_rejected_logprob,
                reference_chosen_logprob,
                reference_rejected_logprob,
                beta,
                reduction,
            )

            assert loss.requires_grad
            if reduction == "mean":
                assert loss.shape == ()
            elif reduction == "sum":
                assert loss.shape == ()
            elif reduction == "none":
                assert loss.shape == (2,)

            assert (loss > 0).all()

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test DPO"
    )
    @pytest.mark.parametrize("reward_threshold", [0.0, "mean", "median"])
    def test_dpo_acceptance_reward_selector(
        self, preference_data, reward_threshold, policy_train
    ):
        from torchrl.data import LazyStackStorage, ReplayBuffer
        from torchrl.data.llm.acceptance import (
            AcceptanceRewardSampler,
            AcceptanceRewardSelector,
        )

        policy_train, tokenizer = policy_train
        rb = ReplayBuffer(
            storage=LazyStackStorage(4),
            transform=AcceptanceRewardSelector(
                reward_threshold=reward_threshold, total_dialog_turns=2
            ),
            sampler=AcceptanceRewardSampler(total_dialog_turns=2),
        )

        td = preference_data.copy()
        del td["next", "is_chosen"]
        td["text"] = NonTensorStack(
            *[
                h.apply_chat_template(
                    tokenizer=tokenizer,
                    chat_template_name="qwen",
                    add_generation_prompt=True,
                )
                for h in td["history"][..., 0].unbind(0)
            ]
        ).unsqueeze(-1)

        assert len(td["text"]) == 4
        assert td["text"][0] == td["text"][1]
        assert td["text"][2] == td["text"][3]
        assert td.shape == (4, 1)
        rb.extend(td)
        assert len(rb) == 2
        data = rb.sample(10)
        assert data["next", "is_chosen"].shape == (2, 10, 1, 1)
        assert data["next", "is_chosen"][0].all()
        assert not data["next", "is_chosen"][1].any()

        data = rb[:]
        assert (
            data["next", "is_chosen"].squeeze()
            == torch.tensor([True, False, True, False]).view(2, 2)
        ).all()

        # Test loss execution


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
