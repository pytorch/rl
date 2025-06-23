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

from tensordict import lazy_stack, set_capture_non_tensor_stack, TensorDict
from torchrl.data import History, LazyStackStorage, ReplayBuffer, Unbounded
from torchrl.envs import Transform
from torchrl.envs.llm import LLMEnv, RetrieveKL
from torchrl.envs.llm.datasets.gsm8k import GSM8KEnv
from torchrl.envs.llm.transforms.kl import RetrieveLogProb
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from torchrl.objectives import ClipPPOLoss
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
    @pytest.fixture(scope="class")
    def actor_network_from_text(self):
        return self._actor_network(from_text=True)

    @pytest.fixture(scope="class")
    def actor_network_from_tokens(self):
        return self._actor_network(from_text=False)

    def _actor_network(self, from_text=True, policy_train_from_text=False):
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
            model, tokenizer=tokenizer, generate=False, from_text=policy_train_from_text
        )
        for p in policy_train.parameters():
            assert p.requires_grad
        return policy_train, policy_inference, tokenizer

    @pytest.mark.skipif(
        not _has_transformers, reason="transformers lib required to test PPO with LLMs"
    )
    @set_capture_non_tensor_stack(False)
    @pytest.mark.parametrize("from_text", [True, False])
    @pytest.mark.parametrize("cls", [ClipPPOLoss, GRPOLoss])
    def test_hf(
        self, from_text, cls, actor_network_from_text, actor_network_from_tokens
    ):
        if from_text:
            policy_train, policy_inference, tokenizer = actor_network_from_text
        else:
            policy_train, policy_inference, tokenizer = actor_network_from_tokens
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
            assert loss_vals.loss_objective.shape == ()  # Should be a scalar
            assert (
                loss_vals.loss_objective.requires_grad
            )  # Should require gradients for training

    @pytest.fixture(scope="class")
    def data(self):
        data = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        history = History.from_chats(data).unsqueeze(0)
        # Create a trajectory
        td0 = TensorDict(
            done=torch.zeros(1, 1, dtype=torch.bool),
            history=history[..., :2],
            next=TensorDict(
                history=history[..., :3],
                done=torch.zeros(1, 1, dtype=torch.bool),
                # Add a singleton because reward can be cast to account for tokens
                reward=torch.zeros(1, 1, 1, dtype=torch.float),
            ),
            batch_size=(1,),
        )
        td1 = TensorDict(
            done=torch.zeros(1, 1, dtype=torch.bool),
            history=history[..., :3],
            next=TensorDict(
                history=history[..., :4],
                done=torch.ones(1, 1, dtype=torch.bool),
                reward=torch.zeros(1, 1, 1, dtype=torch.float),
            ),
            batch_size=(1,),
        )
        td = lazy_stack([td0, td1], 1)
        return td

    def test_grpo_with_history_singleturn(self, data, actor_network_from_text):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        policy_generate = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            # We have fake data, we just need to add the log-probs
            generate=True,
            from_text=True,
            return_log_probs=False,
            chat_template_name="qwen",
        )
        policy_inference = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            # We have fake data, we just need to add the log-probs
            generate=False,
            from_text=True,
            return_log_probs=True,
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
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            chat_template_name="qwen",
            assistant_only=True,
            use_history=("next", "history"),
        )

        env = GSM8KEnv(
            shuffle=False,
            template_kwargs={
                "add_generation_prompt": True,
                "chat_template_name": "qwen",
            },
            tokenizer=tokenizer,
        )
        # Add a transform to compute the log-probs - inference and ref
        transform_kl = RetrieveKL(
            gen_actor=policy_inference,
            ref_actor=policy_ref,
            tokenizer_kwargs={"chat_template_name": "qwen"},
            tokenizer=tokenizer,
        )
        env.append_transform(transform_kl)
        for p in policy_train.parameters():
            assert p.requires_grad

        reset = env.reset()
        print(reset["text"])
        policy_answer = (
            "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
            "To find the total, I need to add April and May: 48 + 24 = 72. Therefore, Natalia sold 72 clips altogether in April and May.</think>\n<answer>72</answer><|im_end|>"
        )
        reset["text_response"] = [policy_answer]
        s, s_ = env.step_and_maybe_reset(reset)
        assert "kl" in s_
        assert s["next", "reward"].shape[-2] > 1

        data = s
        data["advantage"] = s["next", "reward"] - s["next", "reward"].mean()
        loss = GRPOLoss(
            actor_network=policy_train,
        )
        loss.set_keys(sample_log_prob=("next", "log_probs"))
        loss(data)

    def test_grpo_with_text_text_response(self, actor_network_from_text):
        """Test GRPO using text/text_response strategy for single-turn training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

        # Policy for generation (inference)
        policy_inference = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=True,
            from_text=True,
            return_log_probs=False,
            chat_template_name="qwen",
        )

        # Policy for computing log-probs (training)
        policy_train = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            return_log_probs=True,
            chat_template_name="qwen",
            # Use text/text_response mode (default behavior)
            use_history=False,
        )

        # Create some fake data using text/text_response format
        prompt_text = "What is 2 + 2?"
        response_text = "The answer is 4."

        # Create a simple trajectory with text/text_response
        td = TensorDict(
            text=prompt_text,
            text_response=response_text,
            next=TensorDict(
                reward=torch.randn(1, 1, dtype=torch.float),
                done=torch.ones(1, dtype=torch.bool),
            ),
            batch_size=(1,),
        )

        # Compute log-probs for the response
        with torch.no_grad():
            log_probs_output = policy_train(td)
            assert "log_probs" in log_probs_output
            assert log_probs_output["log_probs"].shape[-1] > 0

        # Add advantage for GRPO
        td["advantage"] = td["next", "reward"] - td["next", "reward"].mean()

        # Create GRPO loss
        loss = GRPOLoss(
            actor_network=policy_train,
        )
        loss.set_keys(sample_log_prob="log_probs")

        # Compute loss
        loss_vals = loss(td)
        assert isinstance(loss_vals, GRPOLossOutput)
        assert loss_vals.loss_objective.shape == ()  # Should be a scalar
        assert (
            loss_vals.loss_objective.requires_grad
        )  # Should require gradients for training

    def test_vllm_with_history(self, actor_network_from_text):
        """Test vLLMWrapper using history mode for multi-turn training."""
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError:
            pytest.skip("vLLM not available")

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token

        # Create a simple vLLM model for testing
        model = LLM("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

        # Policy for computing log-probs with history
        policy_train = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            generate=False,
            from_text=True,
            return_log_probs=True,
            chat_template_name="qwen",
            assistant_only=True,
            use_history=("next", "history"),
        )

        # Create some fake data using history format
        from torchrl.data import History

        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."},
            ]
        ]
        history = History.from_chats(chats)

        # Create a trajectory with history
        td = TensorDict(
            history=history[..., :2],  # First part of conversation
            next=TensorDict(
                history=history,  # Full conversation
                reward=torch.randn(1, 1, dtype=torch.float),
                done=torch.ones(1, dtype=torch.bool),
            ),
            batch_size=(1,),
        )

        # Compute log-probs for all assistant responses
        with torch.no_grad():
            log_probs_output = policy_train(td)
            assert "log_probs" in log_probs_output
            # Should have log-probs for all assistant tokens in the history
            assert log_probs_output["log_probs"].shape[-1] > 0

        # Add advantage for GRPO
        td["advantage"] = td["next", "reward"] - td["next", "reward"].mean()

        # Create GRPO loss
        loss = GRPOLoss(
            actor_network=policy_train,
        )
        loss.set_keys(sample_log_prob="log_probs")

        # Compute loss
        loss_vals = loss(td)
        assert isinstance(loss_vals, GRPOLossOutput)
        assert loss_vals.loss_objective.shape == ()  # Should be a scalar
        assert (
            loss_vals.loss_objective.requires_grad
        )  # Should require gradients for training


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
        pass

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
