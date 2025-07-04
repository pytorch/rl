# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Callable, Literal, Optional

from tensordict import lazy_stack, TensorDictBase
from torchrl._utils import logger as torchrl_logger

from torchrl.data.llm.history import History
from torchrl.envs import Transform
from torchrl.envs.common import EnvBase


class EnhancedReasoningTransform(Transform):
    """An enhanced transform that adds intelligent prompts to improve IFEval response quality.

    This transform analyzes the reward and response quality to add targeted prompts that help
    the LLM improve its reasoning and response format.

    Args:
        cond (Callable[[TensorDictBase], bool]): Condition function that determines when to add prompts.
        strategy (Literal["user_guidance", "format_reminder", "quality_hint", "thinking", "step_by_step"]): 
            The strategy to use for prompting.
        reward_threshold (float): Reward threshold for triggering the transform.
        max_steps (int): Maximum number of steps allowed.
        zero_reward (bool): Whether to zero out the reward when the prompt is added.
        undo_done (bool): Whether to undo the done flag when the prompt is added.
    """

    # Different prompt strategies for different scenarios
    PROMPT_STRATEGIES = {
        "user_guidance": [
            "I notice your response doesn't follow the required format. Please provide your thinking between <think> and </think> tags, and your final answer between <answer> and </answer> tags.",
            "Your response needs to be structured properly. First think through the problem in <think> tags, then give your answer in <answer> tags.",
            "Please reconsider your response. Remember to use <think> tags for your reasoning and <answer> tags for your final response.",
        ],
        "format_reminder": [
            "Remember to use the correct format: <think>your reasoning</think><answer>your answer</answer>",
            "Please structure your response with <think> and <answer> tags as instructed.",
            "Your response should follow this format: <think>...</think><answer>...</answer>",
        ],
        "quality_hint": [
            "Let me help you improve your response. Think about this more carefully and provide a better answer.",
            "Your response could be better. Take a moment to reconsider and provide a more thoughtful answer.",
            "I think you can do better. Please think through this more carefully.",
        ],
        "thinking": [
            "But wait, let me think about this more carefully...",
            "Actually, let me reconsider this...",
            "Let me think about it step by step...",
            "Wait, I need to double-check my reasoning...",
        ],
        "step_by_step": [
            "Let me break this down step by step and think more carefully...",
            "I should approach this systematically. Let me think through each part...",
            "Let me reconsider this by going through it step by step...",
        ]
    }

    def __init__(
        self,
        cond: Callable[[TensorDictBase], bool],
        strategy: Literal["user_guidance", "format_reminder", "quality_hint", "thinking", "step_by_step"] = "user_guidance",
        reward_threshold: float = 1.0,
        max_steps: int = 3,
        zero_reward: bool = True,
        undo_done: bool = True,
        random_prompt: bool = True,
    ) -> None:
        super().__init__()
        
        self.cond = cond
        self.strategy = strategy
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.zero_reward = zero_reward
        self.undo_done = undo_done
        self.random_prompt = random_prompt

    def _get_prompt(self) -> str:
        """Get the appropriate prompt based on the strategy."""
        prompts = self.PROMPT_STRATEGIES[self.strategy]
        if self.random_prompt:
            import random
            return random.choice(prompts)
        return prompts[0]

    def _analyze_response_quality(self, content: str) -> dict:
        """Analyze the quality of the response to determine the best strategy."""
        analysis = {
            "has_think_tags": "<think>" in content and "</think>" in content,
            "has_answer_tags": "<answer>" in content and "</answer>" in content,
            "proper_format": self._check_proper_format(content),
            "malformed_tags": self._check_malformed_tags(content),
            "incomplete_response": len(content.strip()) < 50,
        }
        
        # Determine the best strategy based on analysis
        if not analysis["has_think_tags"] or not analysis["has_answer_tags"]:
            analysis["recommended_strategy"] = "format_reminder"
        elif analysis["malformed_tags"]:
            analysis["recommended_strategy"] = "format_reminder"
        elif analysis["incomplete_response"]:
            analysis["recommended_strategy"] = "quality_hint"
        else:
            analysis["recommended_strategy"] = "thinking"
            
        return analysis

    def _check_proper_format(self, content: str) -> bool:
        """Check if the response follows the proper IFEval format."""
        # Check for proper tag structure
        think_pattern = r"<think>.*?</think>"
        answer_pattern = r"<answer>.*?</answer>"
        
        has_think = bool(re.search(think_pattern, content, re.DOTALL))
        has_answer = bool(re.search(answer_pattern, content, re.DOTALL))
        
        return has_think and has_answer

    def _check_malformed_tags(self, content: str) -> bool:
        """Check for malformed tags with extra spaces or wrong format."""
        malformed_patterns = [
            r"<\s*think\s*>",  # < think >
            r"<\s*answer\s*>",  # < answer >
            r"</\s*think\s*>",  # </ think >
            r"</\s*answer\s*>",  # </ answer >
        ]
        
        for pattern in malformed_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Process the tensordict and add enhanced prompts based on the condition."""
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
            raise RuntimeError("EnhancedReasoningTransform must be used with a ChatEnv")
        base_env = parent.base_env
        if base_env.input_mode != "history":
            raise RuntimeError(
                "EnhancedReasoningTransform must be used with a ChatEnv in history mode"
            )

        # Check if we should add the prompt
        if self.cond(next_tensordict):
            torchrl_logger.info(f"Adding enhanced reasoning prompt with strategy: {self.strategy}")
            
            history: History = next_tensordict["history"].prompt
            last_turn = history[..., -1]
            
            # Analyze the last response to determine the best strategy
            if self.strategy == "user_guidance":
                # Use user guidance strategy - add as a user message
                prompt = self._get_prompt()
                history = history.append(History(role="user", content=prompt))
                next_tensordict["history"].prompt = history
                
            elif self.strategy == "thinking":
                # Use thinking strategy - add as assistant message
                prompt = self._get_prompt()
                history = history.append(History(role="assistant", content=prompt))
                next_tensordict["history"].prompt = history
                
            else:
                # For other strategies, use user guidance as default
                prompt = self._get_prompt()
                history = history.append(History(role="user", content=prompt))
                next_tensordict["history"].prompt = history

            # Undo done flag if requested
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
            torchrl_logger.info("Not adding enhanced reasoning prompt.")
            
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Reset the transform state."""
        return tensordict_reset 