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
from functools import partial

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
from torchrl.data.llm.history import History
from torchrl.envs import StepCounter
from torchrl.envs.llm import (
    as_padded_tensor,
    ChatEnv,
    DataLoadingPrimer,
    GSM8KEnv,
    KLRewardTransform,
    LLMEnv,
    make_gsm8k_env,
    RetrieveKL,
)

from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers import AutoTokenizer

_has_ray = importlib.util.find_spec("ray") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None
_has_datasets = importlib.util.find_spec("datasets") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None
_has_ifeval = (
    _has_datasets
    and (importlib.util.find_spec("langdetect") is not None)
    and (importlib.util.find_spec("nltk") is not None)
    and (importlib.util.find_spec("immutabledict") is not None)
)


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    seed = 2
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yield


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

    @pytest.mark.parametrize("input_mode", ["text", "tokens", "history"])
    def test_chat_env(self, tokenizer, input_mode):
        # Set list to stack for tensordict
        set_list_to_stack(True).set()
        # Initialize environment
        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
            system_prompt="I'm system, do what I want.",
            input_mode=input_mode,
        )
        # Reset environment
        td_reset = TensorDict(
            query=["I'm the user. I'm going to tell you a little about something."],
            batch_size=(1,),
            device=env.device,
        )
        td_reset = env.reset(td_reset)
        # Check history after reset
        if input_mode == "history":
            torchrl_logger.info(f'{td_reset["history"].prompt.content=}')
            assert len(td_reset["history"][0].prompt.content) == 2
            assert (
                td_reset["history"][0].prompt[0].content
                == "I'm system, do what I want."
            )
            assert td_reset["history"][0].prompt[1].content.startswith("I'm the user.")
            assert td_reset["history"][0].prompt.role == ["system", "user"]
        elif input_mode == "tokens":
            torchrl_logger.info(f'{td_reset["tokens"].prompt=}')
        elif input_mode == "text":
            torchrl_logger.info(f'{td_reset["text"].prompt=}')
        # Check text after reset
        expected_text = "<|im_start|>system\nI'm system, do what I want.<|im_end|>\n<|im_start|>user\nI'm the user. I'm going to tell you a little about something.<|im_end|>\n<|im_start|>assistant\n"
        if input_mode in ("text",):
            assert td_reset["text"][0].prompt == expected_text
        # Take step in environment
        if input_mode == "history":
            td_reset["history"].response = History(
                content="This is the action from the assistant!", role="assistant"
            ).view(1, 1)
            td_reset["history"].full = td_reset["history"].prompt.extend(
                td_reset["history"].response, dim=-1
            )
            td_action = td_reset
        elif input_mode == "tokens":
            td_reset["tokens"][0].response = tokenizer.encode(
                "This is the action from the assistant!<|im_end|>"
            )
            td_action = td_reset
        elif input_mode == "text":
            td_reset["text"].response = [
                "This is the action from the assistant!<|im_end|>"
            ]
            td_reset["text"].full = [
                td_reset["text"][0].prompt
                + "This is the action from the assistant!<|im_end|>"
            ]
            td_action = td_reset
        td_next = env.step(td_action)
        if input_mode == "history":
            # Check history after step
            assert len(td_next["next", "history"][0].prompt.content) == 3
            assert (
                td_next["next", "history"][0].prompt[0].content
                == "I'm system, do what I want."
            )
            assert (
                td_next["next", "history"][0]
                .prompt[1]
                .content.startswith("I'm the user.")
            )
            assert (
                td_next["next", "history"][0].prompt[2].content
                == "This is the action from the assistant!"
            )
            assert td_next["next", "history"][0].prompt.role == [
                "system",
                "user",
                "assistant",
            ]
        if input_mode in ("text",):
            # Check text after step
            expected_text = "<|im_start|>system\nI'm system, do what I want.<|im_end|>\n<|im_start|>user\nI'm the user. I'm going to tell you a little about something.<|im_end|>\n<|im_start|>assistant\nThis is the action from the assistant!<|im_end|>"
            assert td_next["next", "text"][0].prompt == expected_text


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

    @pytest.mark.parametrize("ray_backend", [True, False], ids=["ray", "local"])
    def test_gsm8kenv(self, ray_backend):
        if not _has_ray and ray_backend:
            pytest.skip("Ray not available")
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = GSM8KEnv(
            tokenizer=tokenizer, apply_template=True, ray_backend=ray_backend
        )
        # env.check_env_specs(break_when_any_done="both")
        r = env.reset()
        assert "history" in r
        assert r["history"].prompt.shape == (1, 2)
        r = r.clone()
        response = "<think>First, calculate the total number of snakes in the breeding balls. There are 3 breeding balls with 8 snakes each, so 3 * 8 = 24 snakes. Next, calculate the number of snakes in the additional pairs. There are 6 pairs of snakes, and each pair has 2 snakes, so 6 * 2 = 12 snakes. Finally, add the number of snakes from the breeding balls and the additional pairs: 24 + 12 = 36 snakes.</think> <answer>Mary saw a total of 36 snakes.</answer><|im_end|>"
        text = (
            r["history"]
            .prompt[0]
            .apply_chat_template(tokenizer=tokenizer, add_generation_prompt=True)
            + response
        )
        history_full = History.from_text(text).unsqueeze(0)
        assert history_full.shape[-1] == 3
        r["history"].full = history_full
        s = env.step(r)
        assert s["next", "reward"] >= 10
        assert s["next", "done"].all()


@pytest.mark.skipif(not _has_ifeval, reason="requires IFEval libs")
class TestIFEvalEnv:
    def test_ifeval(self):
        import torch
        from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
        from transformers import AutoTokenizer

        torch.manual_seed(0)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = IFEvalEnv(apply_template=True, tokenizer=tokenizer, input_mode="history")
        torchrl_logger.info(env.reset())
        r = env.reset()
        r["history"].full = History.from_text(
            r["history"]
            .prompt[0]
            .apply_chat_template(tokenizer=tokenizer, add_generation_prompt=True)
            + """<think>
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
        ).unsqueeze(0)
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
            tokenizer=tokenizer,
            input_mode="history",
        )
        env = base_env.append_transform(PythonInterpreter())
        r = env.reset(
            TensorDict(
                {base_env.data_key: ["This is the user prompt"]}, batch_size=(1,)
            )
        )
        rc = r.clone()
        h = r["history"].prompt
        history_from_text = h.apply_chat_template(tokenizer=tokenizer)
        assert history_from_text == [
            "<|im_start|>system\nI'm the system, do as I say<|im_end|>\n<|im_start|>user\nThis is the user prompt<|im_end|>\n<|im_start|>assistant\n"
        ]
        r["history"].full = h.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\nprint(1 + 1)\n```",
            ).view(1, 1),
            dim=-1,
        )
        s = env.step(r)
        history_str = s["next", "history"].prompt.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=True
        )
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
            "    <|im_start|>user\n"
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
        r["history"].full = h.extend(
            History(
                role="assistant",
                content="Here is a response without a python code to execute.",
            ).view(1, 1),
            dim=-1,
        )
        s = env.step(r)
        history_str = s["next", "history"].prompt.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=True
        )
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
            "    <|im_start|>assistant\n"
            "Here is a response without a python code to execute.<|im_end|>\n"
            "    <|im_start|>assistant\n"
        ]

    def test_python_interpreter_persistent(self):
        pass

        from torchrl.envs.llm.transforms import PythonInterpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        env = ChatEnv(
            batch_size=(1,),
            system_prompt="I'm the system, do as I say",
            tokenizer=tokenizer,
            input_mode="history",
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(
            TensorDict({env.data_key: ["This is the user prompt"]}, batch_size=(1,))
        )
        r["history"].full = r["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\na=1\n```",
            ).view(1, 1),
            dim=-1,
        )
        s, s_ = env.step_and_maybe_reset(r)
        s_["history"].full = s_["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\na+=1\nassert a == 2\n```",
            ).view(1, 1),
            dim=-1,
            inplace=False,
        )
        s, s_ = env.step_and_maybe_reset(s_)
        response = s_["history"].prompt.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=True
        )

        assert response == [
            "<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            "a=1\n"
            "```<|im_end|>\n"
            "    <|im_start|>user\n"
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
            "    <|im_start|>user\n"
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
            tokenizer=tokenizer,
            input_mode="history",
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(
            TensorDict({env.data_key: ["This is the user prompt"]}, batch_size=(1,))
        )
        r["history"].full = r["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\nraise ValueError('This is an error')\n```",
            ).view(1, 1),
            dim=-1,
        )
        s, s_ = env.step_and_maybe_reset(r)
        s_["history"].full = s_["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\na=1\nassert a == 1\n```",
            ).view(1, 1),
            dim=-1,
        )
        s, s_ = env.step_and_maybe_reset(s_)
        assert re.match(
            s_["history"].prompt.apply_chat_template(
                tokenizer=tokenizer, add_generation_prompt=True
            )[0],
            r"<|im_start|>system\n"
            "I'm the system, do as I say<|im_end|>\n"
            "<|im_start|>user\n"
            "This is the user prompt<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Here is a python code to execute:\n"
            "```python\n"
            'raise ValueError("This is an error")\n'
            "```<|im_end|>\n"
            "    <|im_start|>user\n"
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
            "    <|im_start|>user\n"
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
            tokenizer=tokenizer,
        )
        env = env.append_transform(PythonInterpreter(persistent=True))
        r = env.reset(
            TensorDict({env.data_key: ["This is the user prompt"]}, batch_size=(1,))
        )
        r["history"].full = r["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\na = [0]\n```",
            ).view(1, 1),
            dim=-1,
        )
        s, s_ = env.step_and_maybe_reset(r)
        r = env.reset(
            TensorDict({env.data_key: ["This is the user prompt"]}, batch_size=(1,))
        )
        r["history"].full = r["history"].prompt.extend(
            History(
                role="assistant",
                content="Here is a python code to execute:\n```python\n# check if a is still defined\nif 'a' in globals():\n    raise RuntimeError('a is still defined')\nelse:\n    print('a is not defined')\n```",
            ).view(1, 1),
            dim=-1,
        )
        s, s_ = env.step_and_maybe_reset(r)
        assert re.match(
            s_["history"].prompt.apply_chat_template(
                tokenizer=tokenizer, add_generation_prompt=True
            )[0],
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
            "    <|im_start|>user\n"
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
        base_env = ChatEnv(
            batch_size=(1,),
            system_prompt="You are a helpful assistant that uses a calculator.",
            tokenizer=tokenizer,
        )
        transform = MCPToolTransform(tools, schemas)
        env = base_env.append_transform(transform)

        # Test single tool call
        td = TensorDict(
            {base_env.data_key: ["Let me calculate 2 + 3"]}, batch_size=(1,)
        )
        td = env.reset(td)
        td["history"].full = td["history"].prompt.extend(
            History(
                role="assistant",
                content='I will help you calculate 2 + 3:\n<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool><|im_end|>',
            ).view(1, 1),
            dim=-1,
        )
        result = env.step(td)

        # Check that the tool was executed and returned correct result
        history = result["next", "history"].prompt
        assert len(history[0]) == 4  # system, user, assistant, tool response
        assert history[0, -1].role == "tool"
        assert "result': 5" in history[0, -1].content

        # Test multiple tool calls in one response
        td = TensorDict(
            {base_env.data_key: ["Calculate 2 + 3 and 4 * 5"]}, batch_size=(1,)
        )
        td = env.reset(td)
        td["history"].full = td["history"].prompt.extend(
            History(
                role="assistant",
                content='I will help you calculate both:\n<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool>\n<tool>calculator\n{"operation": "multiply", "a": 4, "b": 5}</tool><|im_end|>',
            ).view(1, 1),
            dim=-1,
        )
        result = env.step(td)

        # Check that both tools were executed and returned correct results
        history = result["next", "history"].prompt
        assert (
            len(history[0]) == 5
        )  # system, user, assistant, tool response 1, tool response 2
        assert history[0, -2].role == "tool"
        assert history[0, -1].role == "tool"
        assert "result': 5" in history[0, -2].content  # 2 + 3 = 5
        assert "result': 20" in history[0, -1].content  # 4 * 5 = 20

        # Test error handling
        td = TensorDict({base_env.data_key: ["Calculate 2 ? 3"]}, batch_size=(1,))
        td = env.reset(td)
        td["history"].full = td["history"].prompt.extend(
            History(
                role="assistant",
                content='I will try to calculate:\n<tool>calculator\n{"operation": "invalid", "a": 2, "b": 3}</tool><|im_end|>',
            ).view(1, 1),
            dim=-1,
        )
        result = env.step(td)

        # Check that error was handled gracefully
        history = result["next", "history"].prompt
        assert len(history[0]) == 4
        assert history[0, -1].role == "tool"
        assert "failed" in history[0, -1].content
        assert "Unknown operation: invalid" in history[0, -1].content

        # Test invalid JSON
        td = TensorDict({base_env.data_key: ["Calculate something"]}, batch_size=(1,))
        td = env.reset(td)
        td["history"].full = td["history"].prompt.extend(
            History(
                role="assistant",
                content="Let me calculate:\n<tool>calculator\ninvalid json</tool><|im_end|>",
            ).view(1, 1),
            dim=-1,
        )
        result = env.step(td)

        # Check that JSON error was handled gracefully
        history = result["next", "history"].prompt
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
                query=[["Let me calculate 2 + 3"], ["Let me calculate 4 * 5"]],
                batch_size=(2, 1),
            )
            td = env_pool.reset(tdreset)

            # Send async steps to both environments
            td["history"].full = torch.stack(
                [
                    td[0]["history"].prompt.extend(
                        History(
                            role="assistant",
                            content='Let me calculate 2 + 3:\n<tool>calculator\n{"operation": "add", "a": 2, "b": 3}</tool><|im_end|>',
                        ).view(1, 1),
                        dim=-1,
                    ),
                    td[1]["history"].prompt.extend(
                        History(
                            role="assistant",
                            content='Let me calculate 4 * 5:\n<tool>calculator\n{"operation": "multiply", "a": 4, "b": 5}</tool><|im_end|>',
                        ).view(1, 1),
                        dim=-1,
                    ),
                ]
            )
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
            history = all_results["next", "history"].prompt
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
        assert (
            reset[0]["history"]
            .prompt[-1]
            .content.startswith("Natalia sold clips to 48 of her friends in April")
        )
        policy_answer = (
            "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
            "To find the total, I need to add April and May: 48 + 24 = 72. Therefore, Natalia sold 72 clips altogether in April and May.</think>\n<answer>322 clips</answer><|im_end|>"
        )
        reset["history"].full = reset["history"].prompt.extend(
            History(role="assistant", content=policy_answer).view(1, 1), dim=-1
        )
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
            assert s["next", "history"].prompt.shape == (1, 3)
        else:
            assert s["next", "history"].prompt.shape == (1, 4)
        if role == "assistant":
            assert s[0]["next", "history"].prompt[-1].role == "assistant"
        else:
            assert s[0]["next", "history"].prompt[-1].role == "user"

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
        assert (
            reset[0]["history"]
            .prompt[-1]
            .content.startswith("Natalia sold clips to 48 of her friends in April")
        )
        policy_answer = (
            "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
            "To find the total, I need to add April and May: 48 + 24 = 72. Therefore, Natalia sold 72 clips altogether in April and May.</think>\n<answer>72</answer><|im_end|>"
        )
        reset["history"].full = reset["history"].prompt.extend(
            History(role="assistant", content=policy_answer).view(1, 1), dim=-1
        )
        s = env.step(reset)
        assert (s["next", "reward"] != 0).all(), s["next", "reward"]
        assert s[0]["next", "history"].prompt[-1].role == "assistant"
        assert s["next", "done"].all()
        assert len(s[0]["next", "history"].prompt) == 3


class TestChatEnvIntegration:
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

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_datasets, reason="datasets not available")
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "input_mode,compute_reward",
        [["history", True], ["history", False], ["text", False], ["tokens", False]],
        ids=[
            "history_compute_reward",
            "history_no_compute_reward",
            "text_no_compute_reward",
            "tokens_no_compute_reward",
        ],
    )
    def test_chat_env_integration_ifeval(self, compute_reward, pad_output, input_mode):
        """Test that the wrapper works correctly with the ChatEnv."""
        import vllm.envs as envs
        from torchrl.envs.llm import IFEvalEnv

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        policy = vLLMWrapper(
            model="Qwen/Qwen2.5-0.5B",
            tokenizer="Qwen/Qwen2.5-0.5B",
            input_mode=input_mode,
            pad_output=pad_output,
            generate=True,
        )
        env = IFEvalEnv(
            max_steps=1,
            compute_reward=compute_reward,
            input_mode=input_mode,
            tokenizer=policy.tokenizer,
        )
        r = env.reset()
        prompt = None
        if input_mode == "history":
            assert r["history", "prompt"].shape == (1, 2)
        elif input_mode == "text":
            prompt = r["text", "prompt"][0]
        r = policy(r)
        if input_mode == "history":
            assert r["history", "response"].shape == (1, 1)
            assert r["history", "full"].shape == (1, 3)
        elif input_mode == "text":
            assert r["text", "full"][0].startswith(prompt)
        r, r_ = env.step_and_maybe_reset(r)
        if input_mode == "history":
            assert r["next", "history", "prompt"].shape == (1, 3)
            assert r_["history", "prompt"] is not None
            assert r_.get(("history", "response"), as_list=True) is None
            assert r_.get(("history", "full"), as_list=True) is None
        assert r["next", "done"].all()
        r = policy(r_)
        r, r_ = env.step_and_maybe_reset(r)

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_datasets, reason="datasets not available")
    @pytest.mark.parametrize(
        "compute_reward", [False, True], ids=["no_compute_reward", "compute_reward"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    def test_chat_env_integration_gsm8k(self, compute_reward, pad_output, input_mode):
        """Test that the wrapper works correctly with the ChatEnv."""
        import vllm.envs as envs
        from torchrl.envs.llm import GSM8KEnv

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        policy = vLLMWrapper(
            model="Qwen/Qwen2.5-0.5B",
            tokenizer="Qwen/Qwen2.5-0.5B",
            input_mode=input_mode,
            pad_output=pad_output,
            generate=True,
        )
        env = GSM8KEnv(
            max_steps=1,
            compute_reward=compute_reward,
            input_mode=input_mode,
            tokenizer=policy.tokenizer,
        )
        r = env.reset()
        prompt = None
        if input_mode == "history":
            assert r["history", "prompt"].shape == (1, 2)
        elif input_mode == "text":
            prompt = r["text", "prompt"][0]
        r = policy(r)
        if input_mode == "history":
            assert r["history", "response"].shape == (1, 1)
            assert r["history", "full"].shape == (1, 3)
        elif input_mode == "text":
            assert r["text", "full"][0].startswith(prompt)
        r, r_ = env.step_and_maybe_reset(r)
        if input_mode == "history":
            assert r["next", "history", "prompt"].shape == (1, 3)
            assert r_["history", "prompt"] is not None
            assert r_.get(("history", "response"), as_list=True) is None
            assert r_.get(("history", "full"), as_list=True) is None
        assert r["next", "done"].all()
        r = policy(r_)
        r, r_ = env.step_and_maybe_reset(r)

    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize("ref_input_mode", ["tokens"], ids=["tokens"])
    @pytest.mark.parametrize(
        "env_class", ["GSM8KEnv", "IFEvalEnv"], ids=["gsm8k", "ifeval"]
    )
    def test_chat_env_kl(
        self,
        transformers_instance,
        vllm_instance,
        pad_output,
        ref_input_mode,
        env_class,
    ):
        """Test that the wrapper works correctly with the ChatEnv."""
        import vllm.envs as envs
        from torchrl.envs.llm import GSM8KEnv, IFEvalEnv

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        vllm_model, vllm_tokenizer = vllm_instance
        tf_model, tf_tokenizer = transformers_instance

        # a policy
        policy = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode="history",
            generate=True,
            pad_output=pad_output,
        )
        ref_model = TransformersWrapper(
            tf_model,
            tokenizer=tf_tokenizer,
            input_mode="tokens",
            # TODO: check that generate=True causes an error
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
        )

        if env_class == "GSM8KEnv":
            env = GSM8KEnv(max_steps=10, num_envs=3, input_mode="history")
        elif env_class == "IFEvalEnv":
            env = IFEvalEnv(max_steps=10, num_envs=3, input_mode="history")
        else:
            raise ValueError(f"Invalid environment class: {env_class}")
        env = env.append_transform(KLRewardTransform(ref_model))
        r = env.rollout(1, policy)
        reward = r.get(("next", "reward"), as_list=not pad_output)
        assert reward is not None
        if pad_output:
            assert reward.shape[0] == 3
            assert reward.shape[1] == 1
            assert reward.shape[2] > 1
            assert reward.shape[3] == 1
        else:
            assert len(reward) == 3
            for r in reward:
                assert r.shape[0] == 1
                assert r.shape[1] > 1
                assert r.shape[2] == 1

    @pytest.mark.parametrize(
        "env_class", ["GSM8KEnv", "IFEvalEnv"], ids=["gsm8k", "ifeval"]
    )
    def test_retrievekl_transform(
        self, transformers_instance, vllm_instance, env_class
    ):
        """Test that the RetrieveKL transform works correctly."""
        from torchrl.collectors.llm.base import LLMCollector
        from torchrl.envs.llm import GSM8KEnv, IFEvalEnv

        model, tokenizer = transformers_instance
        vllm_model, vllm_tokenizer = vllm_instance
        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            pad_output=True,
        )
        if env_class == "GSM8KEnv":
            env = GSM8KEnv(max_steps=1, num_envs=3)
        elif env_class == "IFEvalEnv":
            env = IFEvalEnv(max_steps=1, num_envs=3)
        else:
            raise ValueError(f"Invalid environment class: {env_class}")
        env = env.append_transform(RetrieveKL("from_collector", ref_model))
        c = LLMCollector(
            env,
            policy_factory=partial(
                vLLMWrapper,
                vllm_model,
                tokenizer=vllm_tokenizer,
                input_mode="history",
                generate=True,
                pad_output=True,
            ),
            dialog_turns_per_batch=6,
        )
        for d in c:
            assert ("history", "full") in d
            assert ("next", "history", "prompt") in d
            break
        return


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
