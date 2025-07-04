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

from tensordict import lazy_stack, TensorDict
from torchrl.data import History, LazyStackStorage, ReplayBuffer
from torchrl.envs.llm.transforms.kl import RetrieveLogProb
from torchrl.modules.llm import Text, TransformersWrapper, vLLMWrapper
from torchrl.modules.llm.policies.common import ChatHistory, Masks, Tokens
from torchrl.objectives.llm.grpo import MCAdvantage
from torchrl.objectives.llm.sft import SFTLoss

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None
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
                    text=Text(prompt=prompt),
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
        full_text = history.apply_chat_template(
            tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        )
        text_response = [
            txt[len(txt_start) :] for txt, txt_start in zip(full_text, text)
        ]
        td = TensorDict(
            text=Text(prompt=text, response=text_response, full=full_text),
            history=ChatHistory(
                full=history, prompt=history[..., :-1], response=history[..., -1:]
            ),
            next=TensorDict(
                reward=torch.randn(2, 1),
                done=torch.zeros(2, dtype=torch.bool),
                history=ChatHistory(prompt=history),
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
            chat_template_name="qwen",
            input_mode="history",
            pad_output=False,
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
                return_log_probs=True,
                chat_template_name="qwen",
                input_mode="history",
                pad_output=False,
            )
            transform = RetrieveLogProb(
                policy_ref,
                assistant_only=True,
                tokenizer_kwargs={"chat_template_name": "qwen"},
                tokenizer=tokenizer,
                log_probs_key=("ref_log_prob", "full"),
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
        from torchrl.data.llm.history import _CHAT_TEMPLATES
        from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]

        model = OPTForCausalLM(OPTConfig()).eval()
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            chat_template_name="qwen",
        )
        policy_ref = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            return_log_probs=True,
            chat_template_name="qwen",
        )
        transform = RetrieveLogProb(
            policy_ref,
            assistant_only=True,
            tokenizer_kwargs={"chat_template_name": "qwen"},
            tokenizer=tokenizer,
            log_probs_key=("ref_log_prob", "full"),
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


class TestGRPOLossIntegration:
    """Test GRPOLoss integration with the new distribution methods."""

    @pytest.fixture(scope="module")
    def transformers_instance(self):
        """Create transformers model and tokenizer for testing."""
        if not _has_transformers:
            pytest.skip("transformers not available")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @pytest.fixture(scope="module")
    def vllm_instance(self):
        """Create vLLM model and tokenizer for testing."""
        if not _has_vllm:
            pytest.skip("vllm not available")

        import vllm.envs as envs
        from transformers import AutoTokenizer
        from vllm import LLM

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        try:
            model = LLM("Qwen/Qwen2.5-0.5B")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Failed to load vLLM model: {e}")

    @pytest.fixture(scope="module")
    def sample_tokens(self, vllm_instance):
        """Create sample tokens for testing."""
        model, tokenizer = vllm_instance
        text = [
            "Are you happy? Say yes or no.",
            "Explain the difference between a cat and a dog. Be very detailed.",
        ]
        tokenized = tokenizer(
            text, return_tensors="pt", padding=True, padding_side="left"
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    @pytest.fixture(scope="module")
    def sample_text(self):
        """Create sample text for testing."""
        return [
            "Are you happy? Say yes or no.",
            "Explain the difference between a cat and a dog. Be very detailed.",
        ]

    @pytest.fixture(scope="module")
    def sample_history(self):
        """Create sample conversation history for testing."""
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Are you happy? Say yes or no."},
            ],
            [
                {
                    "role": "system",
                    "content": "You are a very helpful assistant, but more handsome.",
                },
                {
                    "role": "user",
                    "content": "Explain the difference between a cat and a dog. Be very detailed.",
                },
            ],
        ]
        return History.from_chats(chats)

    @pytest.fixture(scope="module")
    def sample_history_assistant(self):
        """Create sample conversation history for testing."""
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Are you happy? Say yes or no."},
                {"role": "assistant", "content": "Yes."},
            ],
            [
                {
                    "role": "system",
                    "content": "You are a very helpful assistant, but more handsome.",
                },
                {
                    "role": "user",
                    "content": "Explain the difference between a cat and a dog. Be very detailed.",
                },
                {
                    "role": "assistant",
                    "content": "A cat is a small animal that meows, while a dog is a larger animal that barks.",
                },
            ],
        ]
        return History.from_chats(chats)

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.parametrize("masking_strategy", ["sft", "rlhf"])
    def test_grpo_loss_with_transformers(
        self,
        vllm_instance,
        transformers_instance,
        sample_history,
        sample_tokens,
        masking_strategy,
    ):
        """Test GRPOLoss with vLLM wrapper and different masking strategies."""
        from torchrl.objectives.llm.grpo import GRPOLoss

        model, tokenizer = transformers_instance
        vllm_model, vllm_tokenizer = vllm_instance

        # Use tokens input mode for SFT, history for RLHF/generic
        if masking_strategy == "sft":
            input_mode = "tokens"
            input_ids, attention_mask = sample_tokens
            input_data = {
                "tokens": Tokens(prompt=input_ids),
                "masks": Masks(all_attention_mask=attention_mask),
            }
        else:
            input_mode = "history"
            input_data = {"history": ChatHistory(prompt=sample_history)}

        wrapper_gen = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            generate=True,
            return_log_probs=True,
            pad_output=True,
            generate_kwargs={"max_tokens": 10},
        )

        # Create test data with advantage and correct batch size
        td = TensorDict(input_data, batch_size=(2,)).to_lazystack(0)
        td = wrapper_gen(td)
        # use a shape that can be broadcast
        td["advantage"] = torch.randn(2, 1, 1)

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            return_log_probs=True,
            pad_output=True,
        )

        # Create GRPOLoss with specified masking strategy
        loss_fn = GRPOLoss(
            actor_network=wrapper,
            masking_strategy=masking_strategy,
        )

        # This should work without shape mismatch errors
        try:
            result = loss_fn(td)
            assert result is not None
        except ValueError as e:
            if "Shape mismatch" in str(e):
                # This is expected if the advantage shape doesn't match the log-prob shape
                # due to different masking strategies
                assert masking_strategy in str(e)
            else:
                raise


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
