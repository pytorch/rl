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
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from torchrl.modules.llm.policies.common import ChatHistory, Masks, Text, Tokens
from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage
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
                    query=prompt,  # MCAdvantage expects "query" key, not "text"
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


# Mock infrastructure moved to conftest.py


def _mock_data_grpo(
    vocab_size: int, device: torch.device | str = "cpu"
) -> TensorDict:
    from transformers import AutoTokenizer

    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    prompt = History(
        role=["system", "user"],
        content=["You are a useful assistant.", "What is 2+2?"],
        batch_size=(2,),
        device=device,
    )
    response = History(
        role=["assistant"],
        content=["2 + 2 = 4."],
        batch_size=(1,),
        device=device,
    )
    full_history = prompt.extend(response, inplace=False)
    history = ChatHistory(
        prompt=prompt,
        response=response,
        full=full_history,
        device=device,
    )
    batch_size = 1

    # Expand history to match batch size before getting tokens
    history = history.expand((batch_size,))
    next_history = ChatHistory(
        prompt=full_history,
        device=device,
    )
    next_history = next_history.expand((batch_size,))

    # Now get tokens from the expanded history objects
    tokens_full = history.to_tokens(tokenizer)
    next_tokens = next_history.to_tokens(tokenizer)

    # Get the actual sequence length from the tokens
    # tokens_full has structure with "full" key containing the actual tokens
    # We need to get the padded version to know the actual length
    tokens_input_ids = tokens_full.get(
        "full", as_padded_tensor=True, padding_side="left", padding_value=0
    )
    seq_len = tokens_input_ids.shape[-1]

    # Create tensors with proper shapes
    reward = torch.randn(batch_size, seq_len, 1, device=device)
    done = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device)
    advantage = torch.randn(batch_size, seq_len, 1, device=device)
    log_probs = torch.randn_like(tokens_full, dtype=torch.float32, device=device)

    # Create attention mask (all ones for non-padded tokens)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Import Masks to create proper mask structure
    from tensordict import MetaData
    from torchrl.modules.llm.policies.common import Masks

    masks = Masks(
        all_attention_mask=attention_mask,
        all_assistant_mask=None,  # Will be computed by the wrapper
        padded=MetaData(True),
        device=device,
    )

    data = TensorDict(
        {
            "advantage": advantage,
            "history": history,
            "tokens": tokens_full % vocab_size,
            "masks": masks,
            "next": {
                "history": next_history,
                "tokens": next_tokens % vocab_size,
                "reward": reward,
                "done": done,
            },
            "log_probs": log_probs,
        },
        batch_size=(batch_size,),
    )
    return data


class TestLosses:
    def test_grpo(self, mock_transformer_model):
        """Test GRPO loss computation with mock models."""
        vocab_size = 1024
        device = torch.device("cpu")

        # Create mock model and wrap it
        model = mock_transformer_model(vocab_size=vocab_size, device=device)
        actor_network = TransformersWrapper(
            model,
            generate=False,
            pad_output=True,
            input_mode="history",
        )

        # Create loss module
        loss_fn = GRPOLoss(actor_network)

        # Create fake data
        data = _mock_data_grpo(vocab_size=vocab_size, device=device)

        # Compute loss
        loss_vals = loss_fn(data)

        # Assertions: Check output type and structure
        from torchrl.objectives.llm.grpo import GRPOLossOutput

        assert isinstance(
            loss_vals, GRPOLossOutput
        ), f"Expected GRPOLossOutput, got {type(loss_vals)}"

        # Check that all expected keys are present
        assert hasattr(loss_vals, "loss_objective"), "Missing loss_objective"
        assert hasattr(loss_vals, "clip_fraction"), "Missing clip_fraction"
        assert hasattr(loss_vals, "kl_approx"), "Missing kl_approx"
        assert hasattr(loss_vals, "ESS"), "Missing ESS"
        assert hasattr(loss_vals, "entropy"), "Missing entropy"
        assert hasattr(loss_vals, "loss_entropy"), "Missing loss_entropy"

        # Check tensor shapes (all losses should be scalars after reduction)
        assert (
            loss_vals.loss_objective.shape == ()
        ), f"loss_objective should be scalar, got {loss_vals.loss_objective.shape}"
        assert (
            loss_vals.clip_fraction.shape == ()
        ), f"clip_fraction should be scalar, got {loss_vals.clip_fraction.shape}"
        assert (
            loss_vals.kl_approx.shape == ()
        ), f"kl_approx should be scalar, got {loss_vals.kl_approx.shape}"
        assert (
            loss_vals.ESS.shape == ()
        ), f"ESS should be scalar, got {loss_vals.ESS.shape}"

        # Check that losses are finite
        assert torch.isfinite(loss_vals.loss_objective), "loss_objective is not finite"
        assert torch.isfinite(loss_vals.ESS), "ESS is not finite"

        # Check that clip_fraction is in valid range [0, 1]
        assert (
            0 <= loss_vals.clip_fraction <= 1
        ), f"clip_fraction out of range: {loss_vals.clip_fraction}"


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
                log_probs_full_key=("ref_log_probs", "full"),
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
            log_probs_full_key=("ref_log_probs", "full"),
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


@pytest.mark.slow
@pytest.mark.integration
class TestGRPOLossIntegration:
    """Integration tests for GRPOLoss with real models (vLLM + transformers)."""

    @pytest.fixture(scope="class")
    def transformers_instance(self):
        """Create transformers model and tokenizer for testing."""
        if not _has_transformers:
            pytest.skip("transformers not available")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @pytest.fixture(scope="class")
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

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.parametrize("masking_strategy", ["sft", "rlhf"])
    def test_grpo_loss_with_real_models(
        self,
        vllm_instance,
        transformers_instance,
        masking_strategy,
    ):
        """Test GRPOLoss with vLLM generation and transformers loss computation."""
        from torchrl.objectives.llm.grpo import GRPOLoss

        model, tokenizer = transformers_instance
        vllm_model, vllm_tokenizer = vllm_instance

        # Create sample input based on masking strategy
        if masking_strategy == "sft":
            # Use tokens input mode for SFT
            text = [
                "Are you happy? Say yes or no.",
                "What is 2+2?",
            ]
            tokenized = tokenizer(
                text, return_tensors="pt", padding=True, padding_side="left"
            )
            input_data = {
                "tokens": Tokens(prompt=tokenized["input_ids"]),
                "masks": Masks(all_attention_mask=tokenized["attention_mask"]),
            }
            input_mode = "tokens"
        else:
            # Use history input mode for RLHF
            chats = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Are you happy? Say yes or no."},
                ],
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            ]
            sample_history = History.from_chats(chats)
            input_data = {"history": ChatHistory(prompt=sample_history)}
            input_mode = "history"

        # Generate responses with vLLM
        wrapper_gen = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            generate=True,
            return_log_probs=True,
            pad_output=True,
            generate_kwargs={"max_tokens": 10},
        )

        td = TensorDict(input_data, batch_size=(2,)).to_lazystack(0)
        td = wrapper_gen(td)
        td["advantage"] = torch.randn(2, 1, 1)

        # Compute loss with transformers
        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            return_log_probs=True,
            pad_output=True,
        )

        loss_fn = GRPOLoss(actor_network=wrapper, masking_strategy=masking_strategy)

        # Should successfully compute loss
        result = loss_fn(td)
        assert result is not None
        assert hasattr(result, "loss_objective")
        assert torch.isfinite(result.loss_objective)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
