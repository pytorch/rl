# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import random
import re
import time

import pytest
import torch
from mocking_classes_llm import DummyStrDataLoader, DummyTensorDataLoader

from tensordict import (
    lazy_stack,
    NonTensorData,
    NonTensorStack,
    set_capture_non_tensor_stack,
    set_list_to_stack,
    TensorDict,
)

from torchrl._utils import logger as torchrl_logger
from torchrl.data.llm.chat import History
from torchrl.envs import StepCounter
from torchrl.envs.llm import (
    as_padded_tensor,
    ChatEnv,
    DataLoadingPrimer,
    GSM8KEnv,
    KLRewardTransform,
    LLMEnv,
    make_gsm8k_env,
)

from torchrl.modules.llm import TransformersWrapper
from transformers import AutoTokenizer

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_datasets = importlib.util.find_spec("datasets") is not None
_has_ifeval = (
    _has_datasets
    and (importlib.util.find_spec("langdetect") is not None)
    and (importlib.util.find_spec("nltk") is not None)
    and (importlib.util.find_spec("immutabledict") is not None)
)


@pytest.fixture(scope="module", autouse=True)
def list_to_stack_fixture():
    import tensordict

    with tensordict.set_list_to_stack(True):
        yield
    return


@pytest.fixture(scope="session", autouse=True)
def set_list_to_stack_for_test():
    with set_list_to_stack(True):
        yield
    return


class TestLLMEnv:
    @pytest.fixture(scope="class", autouse=True)
    def set_capture(self):
        with set_capture_non_tensor_stack(False):
            yield None
        return

    @pytest.mark.skipif(not _has_transformers, reason="test requires transformers")
    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            # TODO: a bit experimental, fails with check_env_specs
            # [False, "as_nested_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_llm_env(
        self, from_text, stack_method, device, dl_batch_size, env_batch_size
    ):
        if from_text:
            primer = DataLoadingPrimer(
                dataloader=DummyStrDataLoader(batch_size=dl_batch_size),
                batch_size=env_batch_size,
            )
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            primer = DataLoadingPrimer(
                dataloader=DummyTensorDataLoader(
                    batch_size=dl_batch_size, padding=True
                ),
                stack_method=stack_method,
                batch_size=env_batch_size,
            )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv(
                from_text=from_text,
                device=device,
                batch_size=primer.batch_size,
            )
        env = env.append_transform(primer)
        if env_batch_size is None:
            assert env.batch_size == torch.Size((dl_batch_size,))
        else:
            if not isinstance(env_batch_size, tuple):
                env_batch_size = (
                    torch.Size(())
                    if env_batch_size == 0
                    else torch.Size((env_batch_size,))
                )
            assert env.batch_size == env_batch_size

        env.check_env_specs(break_when_any_done="both")

    @pytest.mark.skipif(not _has_transformers, reason="test requires transformers")
    @pytest.mark.parametrize("tokenizer", [True, False])
    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    def test_llm_from_dataloader(
        self,
        from_text,
        stack_method,
        device,
        dl_batch_size,
        env_batch_size,
        tokenizer,
    ):
        from transformers import AutoTokenizer

        if tokenizer and from_text:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            tokenizer = None
        if from_text:
            kwargs = {
                "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
            }
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            kwargs = {
                "dataloader": DummyTensorDataLoader(
                    padding=True, batch_size=dl_batch_size
                ),
                "stack_method": stack_method,
            }
        kwargs.update(
            {
                "batch_size": env_batch_size,
                "from_text": from_text,
                "device": device,
                "has_attention": False,
                "tokenizer": tokenizer,
            }
        )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv.from_dataloader(**kwargs)
        if env_batch_size is None:
            assert env.batch_size == torch.Size((dl_batch_size,))
        else:
            if not isinstance(env_batch_size, tuple):
                env_batch_size = (
                    torch.Size(())
                    if env_batch_size == 0
                    else torch.Size((env_batch_size,))
                )
            assert env.batch_size == env_batch_size
        env.check_env_specs(break_when_any_done="both")

        def policy(td):
            if from_text and tokenizer is None:
                if not td.shape:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorData(
                        "<nothing>", device=device
                    )
                else:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorStack(
                        *[
                            NonTensorData("<nothing>", device=device)
                            for _ in range(td.shape[0])
                        ]
                    )
            else:
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (1,), dtype=torch.int64
                )
            return td

        r = env.rollout(10, policy)
        if env.batch_size == ():
            assert r.ndim == 1
            r = r.unsqueeze(0)
        else:
            assert r.ndim == 2
        if from_text and tokenizer is None:
            assert isinstance(r[0, 0][LLMEnv._DEFAULT_STR_KEY], str)
            assert isinstance(r[0, 1][LLMEnv._DEFAULT_STR_KEY], str)
            assert (
                r[0, 0][LLMEnv._DEFAULT_STR_KEY]
                == r[0, 1][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[0, 0][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            ), (
                r[0, 0][LLMEnv._DEFAULT_STR_KEY],
                r[0, 0][LLMEnv._DEFAULT_ACTION_STR_KEY],
                r[0, 0]["next", LLMEnv._DEFAULT_STR_KEY],
                r[0, 1][LLMEnv._DEFAULT_STR_KEY],
            )
            assert (
                r[0, 1][LLMEnv._DEFAULT_STR_KEY]
                == r[0, 2][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[0, 1][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
            assert (
                r[-1, 0][LLMEnv._DEFAULT_STR_KEY]
                == r[-1, 1][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[-1, 0][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
            assert (
                r[-1, 1][LLMEnv._DEFAULT_STR_KEY]
                == r[-1, 2][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[-1, 1][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
        elif tokenizer is None:
            assert (
                r[0, 0][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[0, 1][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[0, 1][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[0, 2][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[-1, 0][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[-1, 1][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[-1, 1][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[-1, 2][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()

    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            # TODO: a bit experimental, fails with check_env_specs
            # [False, "as_nested_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("repeats", [3])
    def test_llm_from_dataloader_repeats(
        self, from_text, stack_method, device, env_batch_size, dl_batch_size, repeats
    ):
        if from_text:
            kwargs = {
                "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
                "repeats": repeats,
            }
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            kwargs = {
                "dataloader": DummyTensorDataLoader(
                    padding=True, batch_size=dl_batch_size
                ),
                "stack_method": stack_method,
                "repeats": repeats,
            }
        kwargs.update(
            {
                "batch_size": env_batch_size,
                "from_text": from_text,
                "device": device,
                "has_attention": False,
            }
        )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv.from_dataloader(**kwargs)
        assert env.transform.repeats == repeats

        max_steps = 3
        env.append_transform(StepCounter(max_steps=max_steps))

        def policy(td):
            if from_text:
                if not td.shape:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = "<nothing>"
                else:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorStack(
                        *["<nothing>" for _ in range(td.shape[0])]
                    )
            else:
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (1,), dtype=torch.int64
                )
            return td

        r = env.rollout(100, policy, break_when_any_done=False)
        # check that r at reset is always the same
        r_reset = r[..., ::max_steps]
        if from_text:
            all_strings = r_reset.view(-1)[LLMEnv._DEFAULT_STR_KEY]
            assert sum(s == all_strings[0] for s in all_strings) == repeats
            assert sum(s == all_strings[repeats] for s in all_strings) == repeats
            assert sum(s == all_strings[repeats * 2] for s in all_strings) == repeats
        else:
            all_tokens = r_reset.view(-1)[LLMEnv._DEFAULT_TOKEN_KEY]
            assert sum((s == all_tokens[0]).all() for s in all_tokens) == repeats
            assert sum((s == all_tokens[repeats]).all() for s in all_tokens) == repeats
            assert (
                sum((s == all_tokens[repeats * 2]).all() for s in all_tokens) == repeats
            )

    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
        ],
    )
    @pytest.mark.parametrize("device", [None])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("repeats", [3])
    @pytest.mark.parametrize(
        "assign_reward,assign_done", [[True, False], [True, True], [False, True]]
    )
    def test_done_and_reward(
        self,
        from_text,
        stack_method,
        device,
        env_batch_size,
        dl_batch_size,
        repeats,
        assign_reward,
        assign_done,
    ):
        with pytest.raises(
            ValueError, match="from_text"
        ) if from_text else contextlib.nullcontext():
            if from_text:
                kwargs = {
                    "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
                    "repeats": repeats,
                    "assign_reward": assign_reward,
                    "assign_done": assign_done,
                }
            else:
                if stack_method is None:
                    stack_method = as_padded_tensor
                kwargs = {
                    "dataloader": DummyTensorDataLoader(
                        padding=True, batch_size=dl_batch_size
                    ),
                    "stack_method": stack_method,
                    "repeats": repeats,
                    "assign_reward": assign_reward,
                    "assign_done": assign_done,
                }
            kwargs.update(
                {
                    "batch_size": env_batch_size,
                    "from_text": from_text,
                    "device": device,
                    "has_attention": False,
                }
            )
            with pytest.warns(UserWarning, match="eos_token_id"):
                env = LLMEnv.from_dataloader(**kwargs)
            # We want to make sure that transforms that rely on the done state work appropriately
            env.append_transform(StepCounter(max_steps=10))

            def policy(td):
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (torch.randint(10, (1,)).item(),), dtype=torch.int64
                )
                return td

            r = env.rollout(100, policy, break_when_any_done=False)
            if assign_done:
                assert "terminated" in r
                assert "done" in r


class TestChatEnv:
    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    def test_chat_env(slef, tokenizer):
        # Set list to stack for tensordict
        set_list_to_stack(True).set()
        # Initialize environment
        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
            apply_template=True,
            system_prompt="I'm system, do what I want.",
        )
        # Reset environment
        td_reset = env.reset(
            TensorDict(
                text=["I'm the user. I'm going to tell you a little about something."],
                batch_size=(1,),
            )
        )
        # Check history after reset
        torchrl_logger.info(f'{td_reset["history"].content=}')
        assert len(td_reset["history"][0].content) == 2
        assert td_reset["history"][0, 0].content == "I'm system, do what I want."
        assert td_reset["history"][0, 1].content.startswith("I'm the user.")
        assert td_reset["history"][0].role == ["system", "user"]
        # Check text after reset
        expected_text = "<|im_start|>system\nI'm system, do what I want.<|im_end|>\n<|im_start|>user\nI'm the user. I'm going to tell you a little about something.<|im_end|>\n<|im_start|>assistant\n"
        assert td_reset["text"][0] == expected_text
        # Take step in environment
        td_action = td_reset.set(
            "text_response", ["This is the action from the assistant!<|im_end|>"]
        )
        td_next = env.step(td_action)
        # Check history after step
        assert len(td_next["next", "history"].content[0]) == 3
        assert td_next["next", "history"][0, 0].content == "I'm system, do what I want."
        assert td_next["next", "history"][0, 1].content.startswith("I'm the user.")
        assert (
            td_next["next", "history"][0, 2].content
            == "This is the action from the assistant!"
        )
        assert td_next["next", "history"][0].role == ["system", "user", "assistant"]
        # Check text after step
        expected_text = "<|im_start|>system\nI'm system, do what I want.<|im_end|>\n<|im_start|>user\nI'm the user. I'm going to tell you a little about something.<|im_end|>\n<|im_start|>assistant\nThis is the action from the assistant!<|im_end|>\n<|im_start|>assistant\n"
        assert td_next["next", "text"][0] == expected_text


@pytest.mark.skipif(not _has_datasets, reason="requires datasets")
class TestGSM8K:
    @pytest.fixture(scope="class")
    def ref_model(self):
        if not _has_transformers:
            yield
        from transformers import AutoTokenizer, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m").eval()

        tokenizer.pad_token = "<|PAD|>"
        tokenizer.padding_side = "left"

        yield model, tokenizer

    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_env(self, n_envs):
        with pytest.warns(UserWarning, match="No tokenizer specified"):
            env = make_gsm8k_env(num_envs=n_envs)
        env.check_env_specs(break_when_any_done="both")
        r = env.rollout(3)
        assert ("next", "reward") not in r
        assert r.numel() == n_envs
        r = env.rollout(3, break_when_any_done=False)
        assert r.numel() == n_envs * 3
        assert ("next", "reward") not in r

    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_env_reward(self, n_envs):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        env = make_gsm8k_env(num_envs=n_envs, tokenizer=tokenizer)
        env.check_env_specs(break_when_any_done="both")
        r = env.rollout(3)
        assert ("next", "reward") in r
        assert r.numel() == n_envs
        r = env.rollout(3, break_when_any_done=False)
        assert r.numel() == n_envs * 3
        assert ("next", "reward") in r
        assert r["next", "reward"].shape == (n_envs, 3, 1, 1)

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_kl_bonus(self, n_envs, ref_model):
        torch.manual_seed(0)
        ref_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with torch.device(ref_device):
            model, tokenizer = ref_model
            ref_model = TransformersWrapper(
                model,
                return_log_probs=True,
                generate=False,
                # In practice, we should have the tokens available
                from_text=False,
                tokenizer=tokenizer,
            )
            policy = TransformersWrapper(
                model,
                return_log_probs=True,
                generate=True,
                from_text=True,
                tokenizer=tokenizer,
                generate_kwargs={"max_new_tokens": 20},
                tokenizer_kwargs={"add_special_tokens": False},
            )

            env = make_gsm8k_env(num_envs=n_envs, tokenizer=tokenizer)
            env.append_transform(
                KLRewardTransform(
                    actor=ref_model,
                    coef=0.1,
                    device=ref_device,
                )
            )
            r = env.rollout(3, policy)
            r = r.view(-1)
            for _r in r.unbind(0):
                assert _r["tokens_response"].shape + (1,) == _r["next", "reward"].shape

    def test_gsm8kenv(self):
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = GSM8KEnv(tokenizer=tokenizer, apply_template=True)
        # env.check_env_specs(break_when_any_done="both")
        r = env.reset()
        assert "history" in r
        assert r["history"].shape == (1, 2)
        assert "text" in r
        r = r.clone()
        response = "<think>First, calculate the total number of snakes in the breeding balls. There are 3 breeding balls with 8 snakes each, so 3 * 8 = 24 snakes. Next, calculate the number of snakes in the additional pairs. There are 6 pairs of snakes, and each pair has 2 snakes, so 6 * 2 = 12 snakes. Finally, add the number of snakes from the breeding balls and the additional pairs: 24 + 12 = 36 snakes.</think> <answer>Mary saw a total of 36 snakes.</answer><|im_end|>"
        r["text_response"] = [response]
        s = env.step(r)
        assert s["next", "reward"] >= 10
        assert s["next", "done"].all()

    def test_gsm8kenv_batch(self):
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = GSM8KEnv(tokenizer=tokenizer, apply_template=True, num_envs=4)
        # env.check_env_specs(break_when_any_done="both")
        r = env.reset()
        assert "history" in r
        assert r["history"].shape == (4, 2)
        assert "text" in r
        r = r.clone()
        response = "<think>First, calculate the total number of snakes in the breeding balls. There are 3 breeding balls with 8 snakes each, so 3 * 8 = 24 snakes. Next, calculate the number of snakes in the additional pairs. There are 6 pairs of snakes, and each pair has 2 snakes, so 6 * 2 = 12 snakes. Finally, add the number of snakes from the breeding balls and the additional pairs: 24 + 12 = 36 snakes.</think> <answer>Mary saw a total of 36 snakes.</answer><|im_end|>"
        r["text_response"] = [response] * 4
        s = env.step(r)
        assert (s["next", "reward"] >= 10).all()
        assert s["next", "done"].all()

        env.rollout(10, break_when_any_done=False)


@pytest.mark.skipif(not _has_ifeval, reason="requires IFEval libs")
class TestIFEvalEnv:
    def test_ifeval(self):
        import torch
        from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
        from transformers import AutoTokenizer

        torch.manual_seed(0)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = IFEvalEnv(apply_template=True, tokenizer=tokenizer)
        torchrl_logger.info(env.reset())
        r = env.reset()
        r.set(
            "text_response",
            [
                """<think>
The task requires crafting a riddle about a 'house' that's not traditionally considered one. The answer must be included, and the response should be at least 400 words with a title wrapped in double angular brackets. Let's start by brainstorming what could be considered a 'house' in a non-traditional sense. Ideas include natural shelters, abstract concepts, or objects that serve a similar purpose to a house.
One potential concept is a "womb," as it provides shelter and housing for a developing being. However, we need to ensure our riddle is engaging, meets the word count requirement, and includes the necessary elements like a title.
Let's construct a narrative around the chosen concept, ensuring it's detailed and follows the required structure.
</think>
<answer>
<<A Shelter Beyond Walls>>
In realms beyond the tangible, where shadows softly fall,
A house exists without walls, standing through it all.
No mortar binds its form, no roof shields from the sky,
Yet within its silent depths, a life begins to sigh.
This enigmatic abode, a sanctuary so fine,
Cradles hope and tender shoots, in an embrace divine.
It's not constructed by hands that build and shape with care,
Nor does it stand amidst landscapes that mortals share.
With gentle touch, it nurtures life's first spark,
A haven of beginnings, where love leaves its mark.
Though unseen by eyes that seek the physical form,
It houses dreams and futures, in a quiet, mystic storm.
What is this house that defies the conventional sight?
A place of genesis, bathed in soft, ethereal light.
As we ponder on this riddle, let's unravel the threads:
A womb, a sanctuary, where new life silently spreads.
It's here that the essence of existence begins to unfold,
A journey inward, where the heart beats young and bold.
Within this hidden chamber, protected from the world's din,
A soul stirs, shaped by whispers, and love poured from within.
The umbilical lifeline, a bridge to the outside world,
Nourishes growth, as moments tick by, unfurled.
Thus, we've discovered a 'house' so uniquely designed,
Not made of earthly materials, yet truly sublime.
It's a testament to nature's craft, intricate and grand,
A shelter beyond walls, cradling life's earliest stand.
And so, the riddle unfolds its layers, revealing the core,
A mystery so profound, yet simple, once seen before.
For in the silence, where life takes its first breath,
Lies a 'house' most sacred, a beginning beneath.
In conclusion, the 'house that is not a house' stands as a metaphor for the womb, a place of origin and nurturing. It's a symbol of protection, love, and the inception of life. Through this riddle, we've explored the idea that 'housing' isn't limited to physical structures but can also encompass the intimate spaces where life begins.
The beauty of this concept lies in its universality and the depth of emotion it evokes. It reminds us of the delicate balance between vulnerability and strength, highlighting the miraculous process of creation and growth.
By embracing such metaphors, we're encouraged to look beyond the obvious and appreciate the myriad ways 'shelter' manifests in our lives. And so, the riddle serves not just as a puzzle to be solved but as a reflection on the profound connections that bind us to the very essence of existence.
</answer><|im_end|>
"""
            ],
        )
        td = env.step(r)
        assert td["next", "ifeval_score"].all()
        assert td.get(("next", "reward")) is not None

        # TODO: To test this, we would need to pass a policy to check_env_specs()
        # env.check_env_specs()


class TestTools:
    @pytest.mark.skipif(not _has_transformers, reason="requires transformers")
    def test_python_interpreter_single_batch(self):
        from torchrl.envs.llm.transforms import PythonInterpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        base_env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm the system, do as I say",
            apply_template=True,
            tokenizer=tokenizer,
        )
        env = base_env.append_transform(PythonInterpreter())
        r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        rc = r.clone()
        h = r["history"]
        history_from_text = h.apply_chat_template(tokenizer=tokenizer)
        assert history_from_text == [
            "<|im_start|>system\nI'm the system, do as I say<|im_end|>\n<|im_start|>user\nThis is the user prompt<|im_end|>\n<|im_start|>assistant\n"
        ]
        r["text_response"] = [
            """Here is a python code to execute:
```python
print(1 + 1)
```<|im_end|>\n
"""
        ]
        s = env.step(r)
        history_str = s["next", "history"].apply_chat_template(tokenizer=tokenizer)
        assert history_str == [
            "<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "print(1 + 1)\n"
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 executed successfully:\n"
            "2\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        ]
        history_from_text = History.from_text(history_str, chat_template_name="qwen")
        assert (
            history_from_text
            == lazy_stack(
                [
                    History(role="system", content="I'm the system, do as I say"),
                    History(role="user", content="This is the user prompt"),
                    History(
                        role="assistant",
                        content="Here is a python code to execute:\n```python\nprint(1 + 1)\n```",
                    ),
                    History(
                        role="user",
                        content="<tool_response>\nCode block 1 executed successfully:\n2\n\n</tool_response>",
                        tool_responses=["Code block 1 executed successfully:\n2\n"],
                    ),
                ]
            ).unsqueeze(0)
        ).all()
        # Check what happens if there is no tool response
        r = rc.clone()
        r["text_response"] = [
            """Here is a response without a python code to execute.<|im_end|>"""
        ]
        s = env.step(r)
        history_str = s["next", "history"].apply_chat_template(tokenizer=tokenizer)
        assert history_str == [
            "<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a response without a python code to execute.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ]

    def test_python_interpreter_persistent(self):
        from torchrl.envs.llm.transforms import PythonInterpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm the system, do as I say",
            apply_template=True,
            tokenizer=tokenizer,
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        r["text_response"] = [
            """Here is a python code to execute:
```python
a=1
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(r)
        s_["text_response"] = [
            """Here is a python code to execute:
```python
a+=1
assert a == 2
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(s_)
        assert s_["history"].apply_chat_template(tokenizer=tokenizer) == [
            "<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "a=1\n"
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 executed successfully:\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "a+=1\n"
            "assert a == 2\n"
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 executed successfully:\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        ]

    def test_python_interpreter_persistent_error(self):
        from torchrl.envs.llm.transforms import PythonInterpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm the system, do as I say",
            apply_template=True,
            tokenizer=tokenizer,
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        r["text_response"] = [
            """Here is a python code to execute:
```python
raise ValueError("This is an error")
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(r)
        s_["text_response"] = [
            """Here is a python code to execute:
```python
a=1
assert a == 1
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(s_)
        assert re.match(
            s_["history"].apply_chat_template(tokenizer=tokenizer)[0],
            r"<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            'raise ValueError("This is an error")\n'
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 failed:\n"
            "Error: This is an error\n"
            "Traceback:\n"
            "Traceback (most recent call last):\n"
            '  File "*.py", '
            "line 12, in run_code\n"
            "    exec(compiled, globals(), locals_dict)\n"
            '  File "<string>", line 1, in <module>\n'
            "ValueError: This is an error\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "a=1\n"
            "assert a == 1\n"
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 executed successfully:\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_python_interpreter_persistent_reset(self):
        from torchrl.envs.llm.transforms import PythonInterpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm the system, do as I say",
            apply_template=True,
            tokenizer=tokenizer,
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        r["text_response"] = [
            """Here is a python code to execute:
```python
a = [0]
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(r)
        r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        r["text_response"] = [
            """Here is a python code to execute:
```python
# check if a is still defined
if "a" in globals():
    raise RuntimeError("a is still defined")
else:
    print("a is not defined")
```<|im_end|>\n
"""
        ]
        s, s_ = env.step_and_maybe_reset(r)
        assert re.match(
            s_["history"].apply_chat_template(tokenizer=tokenizer)[0],
            "<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "# check if a is still defined\n"
            'if "a" in globals():\n'
            '    raise RuntimeError("a is still defined")\n'
            "else:\n"
            '    print("a is not defined")\n'
            "```<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Code block 1 executed successfully:\n"
            "a is not defined\n"
            "\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers")
    def test_mcp_tool_transform(self):
        """Test the MCPToolTransform with a simple calculator tool."""
        from torchrl.envs.llm import ChatEnv
        from torchrl.envs.llm.transforms.tools import MCPToolTransform
        from transformers import AutoTokenizer

        # Define a simple calculator tool
        def calculator(operation: str, a: float, b: float) -> dict:
            if operation == "add":
                return {"result": a + b}
            elif operation == "multiply":
                return {"result": a * b}
            else:
                raise ValueError(f"Unknown operation: {operation}")

        # Define the tool schema
        calculator_schema = {
            "name": "calculator",
            "description": "A simple calculator that can add or multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "multiply"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        }

        # Create tools dictionary
        tools = {"calculator": calculator}
        schemas = {"calculator": calculator_schema}

        # Create environment and transform
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="You are a helpful assistant that uses a calculator.",
            apply_template=True,
            tokenizer=tokenizer,
        )
        transform = MCPToolTransform(tools, schemas)
        env = env.append_transform(transform)

        # Test single tool call
        td = TensorDict({"text": ["Let me calculate 2 + 3"]}, batch_size=(1,))
        td = env.reset(td)
        td["text_response"] = [
            'I will help you calculate 2 + 3:\n<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool><|im_end|>'
        ]
        result = env.step(td)

        # Check that the tool was executed and returned correct result
        history = result["next", "history"]
        assert len(history[0]) == 4  # system, user, assistant, tool response
        assert history[0, -1].role == "tool"
        assert "result': 5" in history[0, -1].content

        # Test multiple tool calls in one response
        td = TensorDict({"text": ["Calculate 2 + 3 and 4 * 5"]}, batch_size=(1,))
        td = env.reset(td)
        td["text_response"] = [
            "I will help you calculate both:\n"
            '<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool>\n'
            '<tool>calculator\n{"operation": "multiply", "a": 4, "b": 5}</tool><|im_end|>'
        ]
        result = env.step(td)

        # Check that both tools were executed and returned correct results
        history = result["next", "history"]
        assert (
            len(history[0]) == 5
        )  # system, user, assistant, tool response 1, tool response 2
        assert history[0, -2].role == "tool"
        assert history[0, -1].role == "tool"
        assert "result': 5" in history[0, -2].content  # 2 + 3 = 5
        assert "result': 20" in history[0, -1].content  # 4 * 5 = 20

        # Test error handling
        td = TensorDict({"text": ["Calculate 2 ? 3"]}, batch_size=(1,))
        td = env.reset(td)
        td["text_response"] = [
            'I will try to calculate:\n<tool>calculator\n{"operation": "invalid", "a": 2, "b": 3}</tool><|im_end|>'
        ]
        result = env.step(td)

        # Check that error was handled gracefully
        history = result["next", "history"]
        assert len(history[0]) == 4
        assert history[0, -1].role == "tool"
        assert "failed" in history[0, -1].content
        assert "Unknown operation: invalid" in history[0, -1].content

        # Test invalid JSON
        td = TensorDict({"text": ["Calculate something"]}, batch_size=(1,))
        td = env.reset(td)
        td["text_response"] = [
            "Let me calculate:\n<tool>calculator\ninvalid json</tool><|im_end|>"
        ]
        result = env.step(td)

        # Check that JSON error was handled gracefully
        history = result["next", "history"]
        assert len(history[0]) == 4
        assert history[0, -1].role == "tool"
        assert "failed" in history[0, -1].content
        assert "Failed to parse tool arguments" in history[0, -1].content

    # Define a tool that waits for a random amount of time
    @classmethod
    def delayed_calculator(cls, operation: str, a: float, b: float) -> dict:
        # Random delay between 100ms and 300ms
        delay = random.uniform(0.1, 0.3)
        time.sleep(delay)
        if operation == "add":
            return {"result": a + b, "delay": delay}
        elif operation == "multiply":
            return {"result": a * b, "delay": delay}
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Define the tool schema
    calculator_schema = {
        "name": "delayed_calculator",
        "description": "A calculator that introduces random delays",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }

    # Create environment factory
    @classmethod
    def make_env(cls):
        from torchrl.envs.llm.transforms.tools import MCPToolTransform

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm a calculator assistant",
            apply_template=True,
            tokenizer=tokenizer,
        )
        tools = {"calculator": cls.delayed_calculator}
        schemas = {"calculator": cls.calculator_schema}
        return env.append_transform(MCPToolTransform(tools, schemas))

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers")
    def test_async_mcp_tools(self):
        """Test async execution of MCP tools in an AsyncEnvPool."""
        from tensordict import TensorDict
        from torchrl.envs import AsyncEnvPool

        # Create async env pool with 2 environments
        env_pool = AsyncEnvPool(
            [self.make_env, self.make_env], backend="multiprocessing"
        )
        try:
            # Reset both environments
            tdreset = TensorDict(
                text=[["Let me calculate 2 + 3"], ["Let me calculate 4 * 5"]],
                batch_size=(2, 1),
            )
            td = env_pool.reset(tdreset)

            # Send async steps to both environments
            td["text_response"] = [
                [
                    'Let me calculate 2 + 3:\n<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool><|im_end|>'
                ],
                [
                    'Let me calculate 4 * 5:\n<tool>calculator\n{"operation": "multiply", "a": 4, "b": 5}</tool><|im_end|>'
                ],
            ]
            env_pool.async_step_send(td)

            # Get results as they complete
            results = env_pool.async_step_recv(min_get=1)  # Get at least one result
            assert len(results) >= 1  # We should get at least one result

            # Get remaining results
            if len(results) < 2:
                remaining = env_pool.async_step_recv()
            else:
                remaining = []

            # Combine results
            all_results = torch.stack(list(results) + list(remaining))

            # Verify results
            history = all_results["next", "history"]
            assert len(history[0, 0]) == 4  # system, user, assistant, tool response
            assert history[0, 0, -1].role == "tool"
            assert any(
                "result': 5" in c for c in history[:, 0, -1].content
            )  # 2 + 3 = 5
            assert any(
                "result': 20" in c for c in history[:, 0, -1].content
            )  # 4 * 5 = 20

        finally:
            env_pool.close()


class TestThinkingPrompt:
    @pytest.fixture(autouse=True, scope="class")
    def base_env(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = GSM8KEnv(shuffle=False, tokenizer=tokenizer, max_steps=10)
        return env

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers")
    @pytest.mark.skipif(not _has_datasets, reason="requires gsm8k")
    @pytest.mark.parametrize(
        "role,edit_last_turn",
        [("assistant", True), ("assistant", False), ("user", False)],
    )
    @pytest.mark.parametrize("zero_reward", [True, False])
    @pytest.mark.parametrize("undo_done", [True, False])
    @pytest.mark.parametrize("random_prompt", [True, False])
    def test_thinking_prompt_wrong_answer(
        self,
        role,
        edit_last_turn,
        zero_reward,
        undo_done,
        random_prompt,
        tmp_path,
        base_env,
    ):
        from torchrl.envs.llm.transforms import AddThinkingPrompt

        if isinstance(base_env.transform[-1], AddThinkingPrompt):
            base_env.transform.pop()
        env = base_env.reset_dataloader()
        env = base_env.append_transform(
            AddThinkingPrompt(
                cond=lambda td: td["reward"] < 50,
                role=role,
                edit_last_turn=edit_last_turn,
                zero_reward=zero_reward,
                undo_done=undo_done,
                random_prompt=random_prompt,
            )
        )
        reset = env.reset()
        assert reset[0]["history"][-1].content.startswith(
            "Natalia sold clips to 48 of her friends in April"
        )
        policy_anser = (
            "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
            "To find the total, I need to add April and May: 48 + 24 = 72. Therefore, Natalia sold 72 clips altogether in April and May.</think>\n<answer>322 clips</answer><|im_end|>"
        )
        reset["text_response"] = [policy_anser]
        s = env.step(reset)
        if zero_reward:
            assert (s["next", "reward"] == 0).all()
        else:
            assert (s["next", "reward"] != 0).all()
        if undo_done:
            assert (s["next", "done"] == 0).all()
        else:
            assert (s["next", "done"] != 0).all()
        if edit_last_turn:
            assert s["next", "history"].shape == (1, 3)
        else:
            assert s["next", "history"].shape == (1, 4)
        if role == "assistant":
            assert s[0]["next", "history", "role"][-1] == "assistant"
        else:
            assert s[0]["next", "history", "role"][-1] == "user"

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers")
    @pytest.mark.skipif(not _has_datasets, reason="requires gsm8k")
    @pytest.mark.parametrize(
        "role,edit_last_turn",
        [("assistant", True), ("assistant", False), ("user", False)],
    )
    @pytest.mark.parametrize("zero_reward", [True, False])
    @pytest.mark.parametrize("undo_done", [True, False])
    @pytest.mark.parametrize("random_prompt", [True, False])
    def test_thinking_prompt_correct_answer(
        self,
        role,
        edit_last_turn,
        zero_reward,
        undo_done,
        random_prompt,
        tmp_path,
        base_env,
    ):
        # checks that if cond returns False, nothing is changed
        from torchrl.envs.llm.transforms import AddThinkingPrompt

        if isinstance(base_env.transform[-1], AddThinkingPrompt):
            base_env.transform.pop()
        env = base_env
        env = env.reset_dataloader()
        env = env.append_transform(
            AddThinkingPrompt(
                cond=lambda td: td["reward"] < 50,
                role=role,
                edit_last_turn=edit_last_turn,
                zero_reward=zero_reward,
                undo_done=undo_done,
                random_prompt=random_prompt,
            )
        )
        reset = env.reset()
        assert reset[0]["history"][-1].content.startswith(
            "Natalia sold clips to 48 of her friends in April"
        )
        policy_anser = (
            "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
            "To find the total, I need to add April and May: 48 + 24 = 72. Therefore, Natalia sold 72 clips altogether in April and May.</think>\n<answer>72</answer><|im_end|>"
        )
        reset["text_response"] = [policy_anser]
        s = env.step(reset)
        assert (s["next", "reward"] != 0).all(), s["next", "reward"]
        assert s[0]["next", "history", "role"][-1] == "assistant"
        assert s["next", "done"].all()
        assert len(s[0]["next", "history", "content"]) == 3


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
