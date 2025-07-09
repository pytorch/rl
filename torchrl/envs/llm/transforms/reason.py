# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Callable, Literal

from tensordict import lazy_stack, TensorDictBase
from torchrl._utils import logger as torchrl_logger

from torchrl.data.llm.history import History
from torchrl.envs import Transform
from torchrl.envs.common import EnvBase


class AddThinkingPrompt(Transform):
    """A transform that adds thinking prompts to encourage the LLM to reconsider its response.

    This transform can either add a new thinking prompt as a separate message or edit the last
    assistant response to include a thinking prompt before the final answer. This is useful for
    training LLMs to self-correct and think more carefully when their initial responses are
    incorrect or incomplete.

    Args:
        cond (Callable[[TensorDictBase], bool], optional): Condition function that determines
            when to add the thinking prompt. Takes a tensordict and returns `True` if the prompt
            should be added.
        prompt (str, optional): The thinking prompt to add. If None, a default prompt is used.
            Defaults to `"But wait, let me think about this more carefully..."`.
        random_prompt (bool, optional): Whether to randomly select from predefined prompts.
            Defaults to `False`.
        role (Literal["user", "assistant"], optional): The role for the thinking prompt.
            If `"assistant"`, the prompt is added to the assistant's response. If `"user"`, it's
            added as a separate user message. Defaults to `"assistant"`.
        edit_last_turn (bool, optional): Whether to edit the last assistant response instead
            of adding a new message. Only works with `role="assistant"`. Defaults to `True`.
        zero_reward (bool, optional): Whether to zero out the reward when the thinking prompt
            is added. If `None`, defaults to the value of `edit_last_turn`. Defaults to the same value as `edit_last_turn`.
        undo_done (bool, optional): Whether to undo the done flag when the thinking prompt
            is added. Defaults to `True`.
        egocentric (bool, optional): Whether the thinking prompt is written from the perspective of the assistant.
            Defaults to `None`, which means that the prompt is written from the perspective of the user if `role="user"`
            and from the perspective of the assistant if `role="assistant"`.

    Examples:
        >>> from torchrl.envs.llm.transforms import AddThinkingPrompt
        >>> from torchrl.envs.llm import GSM8KEnv
        >>> from transformers import AutoTokenizer
        >>> import torch
        >>>
        >>> # Create environment with thinking prompt transform
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = GSM8KEnv(tokenizer=tokenizer, max_steps=10)
        >>> env = env.append_transform(
        ...     AddThinkingPrompt(
        ...         cond=lambda td: td["reward"] < 50,
        ...         role="assistant",
        ...         edit_last_turn=True,
        ...         zero_reward=True,
        ...         undo_done=True
        ...     )
        ... )
        >>>
        >>> # Test with wrong answer (low reward)
        >>> reset = env.reset()
        >>> wrong_answer = (
        ...     "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. "
        ...     "Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
        ...     "To find the total, I need to add April and May: 48 + 24 = 72. "
        ...     "Therefore, Natalia sold 72 clips altogether in April and May.</think>"
        ...     "<answer>322 clips</answer><|im_end|>"
        ... )
        >>> reset["text_response"] = [wrong_answer]
        >>> s = env.step(reset)
        >>> assert (s["next", "reward"] == 0).all()  # Reward zeroed
        >>> assert (s["next", "done"] == 0).all()    # Done undone
        >>> assert s["next", "history"].shape == (1, 3)  # History modified
        >>>
        >>> # Test with correct answer (high reward)
        >>> reset = env.reset()
        >>> correct_answer = (
        ...     "<think>Let me solve this step by step. Natalia sold clips to 48 friends in April. "
        ...     "Then she sold half as many in May. Half of 48 is 24. So in May she sold 24 clips. "
        ...     "To find the total, I need to add April and May: 48 + 24 = 72. "
        ...     "Therefore, Natalia sold 72 clips altogether in April and May.</think>"
        ...     "<answer>72</answer><|im_end|>"
        ... )
        >>> reset["text_response"] = [correct_answer]
        >>> s = env.step(reset)
        >>> assert (s["next", "reward"] != 0).all()  # Reward not zeroed
        >>> assert s["next", "done"].all()           # Done remains True
        >>> assert s["next", "history"].shape == (1, 3)  # History unchanged
    """

    # Predefined thinking prompts
    DEFAULT_PROMPTS_EG = [
        "But wait, let me think about this more carefully...",
        "Actually, let me reconsider this...",
        "But we can do better. Let me think about it step by step...",
        "Wait, I need to double-check my reasoning...",
        "Actually, let me think about it more carefully...",
        "It looks like I made a mistake. Let me think about it step by step...",
    ]
    DEFAULT_PROMPTS_COG = [
        "But wait, think about this more carefully...",
        "Actually, reconsider this...",
        "But we can do better. Let's think about it step by step...",
        "Wait, you need to double-check your reasoning...",
        "Actually, think about it more carefully...",
        "It looks like you made a mistake. Can you see what went wrong? Let's think about it step by step...",
    ]

    def __init__(
        self,
        cond: Callable[[TensorDictBase], bool],
        prompt: str | None = None,
        random_prompt: bool = False,
        role: Literal["user", "assistant"] = "assistant",
        edit_last_turn: bool = True,
        zero_reward: bool | None = None,
        undo_done: bool = True,
        egocentric: bool | None = None,
    ) -> None:
        super().__init__()

        # Set condition and role
        self.cond = cond
        self.role = role
        if egocentric is None:
            egocentric = role == "assistant"
        self.egocentric = egocentric

        # Set the prompt
        if prompt is None:
            prompt = (
                self.DEFAULT_PROMPTS_EG[0]
                if egocentric
                else self.DEFAULT_PROMPTS_COG[0]
            )
        self._prompt = prompt
        self.random_prompt = random_prompt

        # Validate edit_last_turn constraint
        if edit_last_turn and role != "assistant":
            raise ValueError("edit_last_turn can only be used with role='assistant'")
        self.edit_last_turn = edit_last_turn

        # Set zero_reward behavior
        if zero_reward is None:
            zero_reward = edit_last_turn
        self.zero_reward = zero_reward
        self.undo_done = undo_done

    @property
    def prompt(self) -> str:
        if self.random_prompt:
            import random

            return random.choice(
                self.DEFAULT_PROMPTS_EG if self.egocentric else self.DEFAULT_PROMPTS_COG
            )
        return self._prompt

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Process the tensordict and add thinking prompts based on the condition.

        Args:
            tensordict: The current tensordict
            next_tensordict: The next tensordict containing the most recent history and reward

        Returns:
            The modified next_tensordict
        """
        # Handle batch dimensions
        if next_tensordict.batch_dims >= 1:
            ntds = []
            for td, next_td in zip(tensordict.unbind(0), next_tensordict.unbind(0)):
                ntds.append(self._step(td, next_td))
            next_tensordict.update(lazy_stack(ntds))
            return next_tensordict

        # Check that base_env is on history mode
        parent = self.parent
        if parent is None:
            raise RuntimeError("AddThinkingPrompt must be used with a ChatEnv")
        base_env = parent.base_env
        if base_env.input_mode != "history":
            raise RuntimeError(
                "AddThinkingPrompt must be used with a ChatEnv in history mode"
            )

        # Check if we should add the thinking prompt
        if self.cond(next_tensordict):
            torchrl_logger.info("Adding thinking prompt.")
            history: History = next_tensordict["history"].prompt
            last_turn = history[..., -1]

            if self.edit_last_turn:

                # Edit the last assistant response
                content = last_turn.content
                modified_content = self._replace_answer_with_prompt(content)

                # Create new history entry with modified content
                new_turn = History(
                    role="assistant",
                    content=modified_content,
                    batch_size=last_turn.batch_size,
                    device=last_turn.device,
                )

                # Replace the last turn in history
                history = history[..., :-1].append(new_turn)
                next_tensordict["history"].prompt = history

            else:
                # Add a new message
                prompt = self.prompt

                history = history.append(History(role=self.role, content=prompt))
                next_tensordict["history"].prompt = history

            if self.undo_done:
                parent: EnvBase = self.parent
                if parent is not None:
                    done_keys = parent.done_keys
                    for key in done_keys:
                        done = next_tensordict.get(key)
                        if done is not None:
                            next_tensordict.set(key, done.zero_())

            # Zero out reward if requested
            if self.zero_reward:
                parent: EnvBase = self.parent
                if parent is not None:
                    reward_keys = parent.reward_keys
                    for key in reward_keys:
                        reward = next_tensordict.get(key)
                        if reward is not None:
                            next_tensordict.set(key, reward.zero_())
        else:
            torchrl_logger.info("Not adding thinking prompt.")
        return next_tensordict

    def _replace_answer_with_prompt(self, content: str) -> str:
        """Replace the last answer section with a thinking prompt.

        This method uses regex to find and replace the last <answer>...</answer> section
        with the thinking prompt, preserving any content before the answer tag.
        Only the last answer block is replaced to avoid interfering with earlier
        examples or instructions that might contain answer tags.

        Args:
            content: The original content string

        Returns:
            The modified content with the last answer replaced by the thinking prompt
        """
        # Pattern to match <answer>...</answer> with optional EOS token
        # Use non-greedy matching and be more specific about the end
        answer_pattern = r"<answer>.*?</answer>(?:\s*<\|im_end\|>)?"

        # Check if there's an answer tag
        if "<answer>" in content:
            # Find all matches to get the last one
            matches = list(re.finditer(answer_pattern, content, flags=re.DOTALL))

            if matches:
                # Get the last match
                last_match = matches[-1]
                start, end = last_match.span()

                # Replace only the last answer section with the thinking prompt
                prompt = self.prompt
                modified_content = content[:start] + prompt + content[end:]

                # Clean up any trailing whitespace
                modified_content = modified_content.rstrip()

                # Ensure we end with the EOS token if the original content had it
                if content.endswith("<|im_end|>"):
                    modified_content = modified_content.rstrip() + "<|im_end|>"

                # Ensure proper spacing around the prompt
                if not modified_content.endswith(prompt):
                    # If the prompt wasn't properly inserted, append it
                    modified_content = content.rstrip()
                    if modified_content.endswith("<|im_end|>"):
                        modified_content = modified_content[
                            : -len("<|im_end|>")
                        ].rstrip()
                    modified_content = modified_content + "\n\n" + prompt + "<|im_end|>"
            else:
                # No matches found, just append the prompt
                prompt = self.prompt
                modified_content = content.rstrip() + "\n\n" + prompt

        else:
            # No answer tag found, just append the prompt
            prompt = self.prompt
            modified_content = content.rstrip() + "\n\n" + prompt

        return modified_content

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Reset the transform state.

        Args:
            tensordict: The current tensordict
            tensordict_reset: The reset tensordict

        Returns:
            The reset tensordict
        """
        return tensordict_reset
