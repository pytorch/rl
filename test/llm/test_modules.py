# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import pytest
import torch

from mocking_classes import DummyStrDataLoader
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
)
from torchrl.collectors.llm import LLMCollector
from torchrl.data.llm import LLMData
from torchrl.envs.llm import LLMEnv
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers import OPTForCausalLM

_has_transformers = importlib.util.find_spec("transformers")
_has_vllm = importlib.util.find_spec("vllm")


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
class TestLLMActor:
    @pytest.fixture(scope="module")
    def vllm_instance(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("facebook/opt-125m")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def transformers_instance(self):
        from transformers import AutoTokenizer

        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # model = GPT2LMHeadModel(GPT2Config()).eval()
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.fixture(scope="module")
    def transformers_instance_pretrained(self):
        from transformers import AutoTokenizer, OPTForCausalLM

        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # model = GPT2LMHeadModel(GPT2Config())
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.mark.parametrize(
        "from_text, generate, return_log_probs, tokens, attention_mask",
        [
            (True, True, True, None, None),
            (True, True, False, None, None),
            (True, False, None, None, None),
            (
                False,
                True,
                True,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, True, torch.randint(1024, (1, 10)), None),
            (
                False,
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_transformers_wrapper(
        self,
        from_text,
        generate,
        return_log_probs,
        tokens,
        attention_mask,
        transformers_instance,
    ):
        torch.manual_seed(0)

        model, tokenizer = transformers_instance

        m = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=generate,
            return_log_probs=return_log_probs,
        )
        self._run_check(
            m,
            tokens,
            attention_mask,
            generate,
            return_log_probs,
            from_text,
            has_logits=True,
        )

    @pytest.mark.skip_if_nightly
    @pytest.mark.parametrize(
        "from_text, generate, return_log_probs, tokens, attention_mask",
        [
            (True, True, True, None, None),
            (True, True, False, None, None),
            (True, False, None, None, None),
            (
                False,
                True,
                True,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, True, torch.randint(1024, (1, 10)), None),
            (
                False,
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_vllm_wrapper(
        self,
        from_text,
        generate,
        return_log_probs,
        tokens,
        attention_mask,
        vllm_instance,
    ):
        torch.manual_seed(0)

        model = vllm_instance
        m = vLLMWrapper(
            model,
            from_text=from_text,
            generate=generate,
            return_log_probs=return_log_probs,
        )
        self._run_check(
            m,
            tokens,
            attention_mask,
            generate,
            return_log_probs,
            from_text,
            has_logits=False,
        )

    def _make_data(
        self,
        m,
        tokens,
        attention_mask,
        generate,
        from_text,
        has_logits,
        batch_size=1,
        text_response=None,
        tokens_response=None,
    ):
        lp_kwargs = {}
        if from_text:
            if not generate:
                text_response = (
                    NonTensorStack(" and another text that follows")
                    if text_response is None
                    else text_response
                )
                if not isinstance(text_response, NonTensorStack):
                    if isinstance(text_response, list):
                        text_response = NonTensorStack(*text_response)
                    else:
                        text_response = NonTensorStack(text_response)
                lp_kwargs.update({"text_response": text_response})
            tdin = LLMData(
                text=NonTensorStack("Somewhere, I lost"),
                **lp_kwargs,
                batch_size=batch_size,
            )
        else:
            if not generate:
                if tokens_response is None:
                    shape_response = tokens.shape
                    shape_response = shape_response[:-1] + (shape_response[-1] * 2,)
                    tokens_response = torch.randint(1024, shape_response)
                lp_kwargs.update({"tokens_response": tokens_response})
            tdin = LLMData(
                tokens=tokens,
                attention_mask=attention_mask,
                **lp_kwargs,
                batch_size=batch_size,
            )
        return tdin

    def _run_check(
        self,
        m,
        tokens,
        attention_mask,
        generate,
        return_log_probs,
        from_text,
        has_logits,
    ):
        tdin = self._make_data(
            m, tokens, attention_mask, generate, from_text, has_logits
        )
        if from_text and generate:
            assert tdin.text_response is None
        elif from_text and not generate:
            assert tdin.text_response is not None

        tdin.copy()
        td = m(tdin)
        assert td is tdin
        assert isinstance(td, LLMData)
        if from_text and generate:
            assert td.text_response is not None

        # TODO: vLLM may produce an attention mask when hf does not - explore consistency!
        # if generate and (from_text or tdincopy.attention_mask is not None):
        #     assert td.attention_mask is not None, (generate, from_text, tdincopy.attention_mask is not None)
        #     if isinstance(td.attention_mask, torch.Tensor):
        #         assert td.attention_mask.shape == td.tokens.shape
        # else:
        #     assert td.attention_mask is None, (generate, from_text)

        if not generate:
            # logprobs are computed on text response of tokens_response
            assert td.text_response is not None or td.tokens_response is not None
            assert td.log_probs is not None
            if has_logits:
                assert td.logits is not None
        if generate:
            if return_log_probs:
                assert td.log_probs is not None
                assert td.log_probs.shape[-1] == td.tokens_response.shape[-1]
            else:
                assert td.log_probs is None

        # Test the shapes
        assert td.tokens_response is not None, (generate, has_logits, from_text)

        # If from text and not generating, the tokens are not returned for now
        if not (from_text and not generate):
            assert td.tokens_response is not None
            assert td.tokens is not None
            assert td.tokens_response.shape[:-1] == td.tokens.shape[:-1]
            # The convention is that the response only has new tokens
            assert (
                td.tokens_response[..., : td.tokens.shape[-1]]
                != td.tokens[..., : td.tokens_response.shape[-1]]
            ).any(), (generate, from_text)

    @pytest.mark.parametrize(
        "from_text, tokens, attention_mask",
        [
            (
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, torch.randint(1024, (1, 10)), None),
            (True, None, None),
        ],
    )
    def test_transformers_logprobs(
        self, from_text, tokens, attention_mask, transformers_instance
    ):
        torch.manual_seed(0)
        model, tokenizer = transformers_instance

        m_generate = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=True,
            return_log_probs=True,
        )
        m_logprobs = TransformersWrapper(
            model, tokenizer=tokenizer, from_text=from_text, generate=False
        )
        self._check_lps(
            m_generate,
            m_logprobs,
            tokens,
            attention_mask,
            from_text,
            has_logits=False,
        )

    @pytest.mark.skip_if_nightly
    @pytest.mark.parametrize(
        "pad_output, from_text, tokens, attention_mask",
        [
            (True, True, None, None),
            (False, True, None, None),
            (
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_vllm_logprobs(
        self, from_text, tokens, attention_mask, pad_output, vllm_instance
    ):
        torch.manual_seed(0)

        model = vllm_instance
        m_generate = vLLMWrapper(
            model,
            from_text=from_text,
            generate=True,
            return_log_probs=True,
            pad_output=pad_output,
        )
        m_logprobs = vLLMWrapper(
            model, from_text=from_text, generate=False, pad_output=pad_output
        )
        self._check_lps(
            m_generate,
            m_logprobs,
            tokens,
            attention_mask,
            from_text,
            has_logits=False,
            tol=1e-1,
        )

    def _check_lps(
        self,
        model_generate,
        model_logprobs,
        tokens,
        attention_mask,
        from_text,
        has_logits,
        tol=1e-2,
    ):
        # Checks that the log-probs gathered with generate=False equate those with generate=True
        tdin_genetate = self._make_data(
            model_generate, tokens, attention_mask, True, from_text, has_logits
        )
        td_generate = model_generate(tdin_genetate)
        tdin_logprobs = self._make_data(
            model_logprobs,
            tokens,
            attention_mask,
            False,
            from_text,
            has_logits,
            tokens_response=td_generate.tokens_response,
            text_response=td_generate.text_response,
        )
        td_logprobs = model_logprobs(tdin_logprobs)
        assert td_generate.tokens_response.shape == td_logprobs.tokens_response.shape
        assert (td_generate.tokens_response == td_logprobs.tokens_response).all(), (
            td_generate.tokens_response == td_logprobs.tokens_response
        )
        assert td_generate.log_probs.shape == td_generate.tokens_response.shape
        assert td_logprobs.log_probs.shape == td_logprobs.tokens_response.shape
        assert td_logprobs.log_probs.shape == td_generate.tokens_response.shape
        torch.testing.assert_close(
            td_generate.log_probs, td_logprobs.log_probs, rtol=tol, atol=tol
        )

    @pytest.mark.skip_if_nightly
    @pytest.mark.parametrize("pad", [True, False])
    @pytest.mark.parametrize("generate", [True, False])
    @pytest.mark.parametrize("use_tensorclass", [True, False])
    def test_vllm_batch_run(self, pad, generate, use_tensorclass, vllm_instance):
        # Test generate - padding combinations
        policy = vLLMWrapper(
            vllm_instance,
            from_text=True,
            generate=generate,
            return_log_probs=True,
            pad_output=pad,
            generate_kwargs={"max_tokens": 10000},
        )
        if generate:
            data = LazyStackedTensorDict(
                *TensorDict(
                    text=NonTensorStack("a string", "another very long string"),
                    batch_size=[2],
                ).unbind(0)
            )
        else:
            data = LazyStackedTensorDict(
                *TensorDict(
                    text=NonTensorStack("a string", "another very long string"),
                    text_response=NonTensorStack(
                        " is a string", " is still a very long string"
                    ),
                    batch_size=[2],
                ).unbind(0)
            )
        if use_tensorclass:
            data = LLMData.from_tensordict(data)
        output = policy(data)
        try:
            log_probs = output.get("log_probs")
        except Exception:
            log_probs = output.get("log_probs", as_list=True)
        if pad:
            assert isinstance(log_probs, torch.Tensor)
        else:
            assert isinstance(log_probs, list)
        text = output.get("text", as_list=True)
        # TODO: this is not ideal...
        if use_tensorclass:
            assert isinstance(text, list)
        else:
            assert isinstance(text, NonTensorStack)
        text_response = output.get("text_response", as_list=True)
        if use_tensorclass:
            assert isinstance(text_response, list)
        else:
            assert isinstance(text_response, NonTensorStack)
        try:
            tokens_response = output.get("tokens_response")
        except Exception:
            tokens_response = output.get("tokens_response", as_list=True)
        if pad:
            assert isinstance(tokens_response, torch.Tensor)
        else:
            assert isinstance(tokens_response, list)
        try:
            tokens = output.get("tokens")
        except Exception:
            tokens = output.get("tokens", as_list=True)
        if not generate:
            assert tokens is None
        elif pad:
            assert isinstance(tokens, torch.Tensor), tokens
        else:
            assert isinstance(tokens, list)

    @pytest.mark.skip_if_nightly
    @pytest.mark.parametrize("from_text", [True])
    def test_vllm_collection(self, vllm_instance, from_text):
        policy = vLLMWrapper(
            vllm_instance,
            return_log_probs=True,
            generate_kwargs={"max_tokens": 32},
            from_text=from_text in (True, None),
        )
        tokenizer = vllm_instance.get_tokenizer()
        self._run_check_collector(policy, from_text=from_text, tokenizer=tokenizer)

    def test_transformers_collection(self):
        ...

    @classmethod
    def env_constructor(cls, **kwargs):
        def make():
            # if kwargs.get("from_text", True):
            dl = DummyStrDataLoader(batch_size=32)
            # else:
            #     dl = DummyTensorDataLoader(batch_size=32)
            env = LLMEnv.from_dataloader(
                dl,
                batch_size=4,
                repeats=4,
                **kwargs,
            )
            assert env.batch_size == (16,)
            return env

        return make

    def _run_check_collector(self, policy, from_text, tokenizer):
        if from_text is None:
            kwargs = {"eos_token_id": tokenizer.eos_token_id}
        else:
            kwargs = {
                "from_text": from_text,
                "tokenizer": tokenizer,
                "eos_token_id": tokenizer.eos_token_id,
            }
        collector = LLMCollector(
            self.env_constructor(**kwargs),
            policy=policy,
            dialog_turns_per_batch=32,
            total_dialog_turns=128,
        )
        t = 0
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert isinstance(data.reshape(-1).get("text_response"), NonTensorStack)
            # action
            assert "text_response" in data
            assert "tokens_response" in data
            # obs
            assert "text" in data
            assert ("next", "text") in data
            # tokens
            assert "tokens" in data

            t += data.numel()
            assert collector._frames == t
            assert t < 512, t  # assert ("next", "tokens") in data

    @pytest.mark.skip_if_nightly
    def test_vllm_generate_multiple_trajs(self, vllm_instance):
        policy = vLLMWrapper(
            vllm_instance,
            return_log_probs=True,
            generate_kwargs={"n": 10, "max_tokens": 1024},
            inplace=False,
        )
        data = TensorDict(
            text=NonTensorStack("a string", "another very long string"), batch_size=2
        )
        data = policy(data)

    @set_list_to_stack(True)
    @pytest.mark.parametrize("from_text", [True, False])
    @pytest.mark.parametrize("generate", [True, False])
    def test_transformers_long_sequences(
        self, from_text, generate, transformers_instance_pretrained
    ):
        torch.manual_seed(42)
        model, tokenizer = transformers_instance_pretrained
        prompts = [
            "The quick brown fox jumps over the lazy dog.",  # Likely to finish soon
            "Once upon a time in a land far, far away, there was a",  # Likely to continue longer
            "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
        ]
        data = lazy_stack([TensorDict() for _ in range(len(prompts))])
        data["text"] = prompts
        eos_token_id = tokenizer.convert_tokens_to_ids(",")
        if not from_text:
            data["tokens"] = tokenizer(data["text"])["input_ids"]
            data["attention_mask"] = (
                0 * data.get("tokens", as_nested_tensor=True, layout=torch.strided) + 1
            )
        if not generate:
            # we need responses
            responses = prompts[1:] + [" et dolore magna aliqua."]
            data["text_response"] = responses
            if not from_text:
                data["tokens_response"] = tokenizer(data["text_response"])["input_ids"]
        # make sure dimensions are ragged for tokens entries
        if "tokens" in data:
            assert data.get_item_shape("tokens")[-1] == -1
        if "tokens_response" in data:
            assert data.get_item_shape("tokens_response")[-1] == -1
        generate_kwargs = {}
        if generate:
            generate_kwargs = {
                "max_new_tokens": 128,  # Set a reasonable number of new tokens to generate
                "min_length": 20,  # Ensure a minimum length for the generated sequence
                "pad_token_id": tokenizer.pad_token_id,  # Use the tokenizer's pad token
                "forced_eos_token_id": eos_token_id,  # Use comma as an EOS token
            }
        policy = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=generate,
            return_log_probs=True,
            # TODO: use n trajs
            generate_kwargs=generate_kwargs,
        )
        data_policy = policy(data)
        if "tokens" in data_policy:
            assert data_policy.get_item_shape("tokens")[-1] == -1
        if "tokens_response" in data_policy:
            assert (
                data_policy.get_item_shape("tokens_response")[-1] == -1
            )  # TODO: this fails


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
