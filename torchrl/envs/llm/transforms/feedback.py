# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import torch
from tensordict import lazy_stack, NestedKey, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.data.llm import History
from torchrl.envs.transforms.transforms import Transform

if TYPE_CHECKING:
    from transformers import AutoTokenizer


# Default template following the SDPO paper
DEFAULT_TEACHER_TEMPLATE = """{prompt}

{feedback_section}
Correctly solve the original question."""

FEEDBACK_WITH_SOLUTION = """Correct solution:
{successful_rollout}

The following is feedback from your unsuccessful earlier attempt:
{env_output}"""

FEEDBACK_WITHOUT_SOLUTION = """The following is feedback from your unsuccessful earlier attempt:
{env_output}"""

SUCCESS_ONLY = """Correct solution:
{successful_rollout}"""


class AddFeedbackContext(Transform):
    """Adds self-teacher context to tensordict for SDPO training.

    This transform prepares the feedback-augmented context needed for Self-Distillation
    Policy Optimization (SDPO). It constructs a "teacher context" by combining:

    1. The original prompt
    2. Rich feedback from the environment (e.g., runtime errors, test failures)
    3. Optionally, a successful solution from another rollout on the same prompt

    The teacher context follows the template from the SDPO paper::

        User: {original_prompt}
              Correct solution: {successful_rollout}  (if available)
              Feedback from unsuccessful attempt: {env_output}
              Correctly solve the original question.
        Assistant: {original_response}

    This transform can be used in two modes:

    1. **Direct mode**: Apply directly to a tensordict that already contains feedback
    2. **Grouped mode**: When attached to a replay buffer, accumulate rollouts and
       find successful solutions within groups (similar to GRPO grouping)

    Args:
        grpo_size (int | None, optional): If set, accumulate rollouts in groups of this
            size per prompt and use successful rollouts as feedback for failed ones.
            Defaults to ``None`` (direct mode).

    Keyword Args:
        prompt_key (NestedKey): Key for the original prompt. Defaults to ``"query"``.
        response_key (NestedKey): Key for the model's response. Defaults to ``("text", "response")``.
        feedback_key (NestedKey): Key for environment feedback (e.g., error messages).
            Defaults to ``"env_feedback"``.
        reward_key (NestedKey): Key for reward signal (used to identify successful rollouts).
            Defaults to ``("next", "reward")``.
        done_key (NestedKey): Key for done signal. Defaults to ``("next", "done")``.
        teacher_context_key (NestedKey): Key where the teacher context will be written.
            Defaults to ``"teacher_context"``.
        success_threshold (float): Reward threshold above which a rollout is considered
            successful. Defaults to ``0.5``.
        include_response_in_context (bool): Whether to include the original response
            in the teacher context. Defaults to ``True``.
        template (str | None): Custom template for constructing teacher context.
            If None, uses the default SDPO template. Defaults to ``None``.
        tokenizer (AutoTokenizer | None): Tokenizer for handling chat templates.
            Defaults to ``None``.
        verbose (bool): Whether to print verbose information. Defaults to ``False``.

    Example:
        >>> # Direct mode: apply to tensordict with feedback
        >>> transform = AddFeedbackContext()
        >>> td["env_feedback"] = "RuntimeError: division by zero at line 73"
        >>> td_with_context = transform(td)
        >>> # td_with_context["teacher_context"] now contains the augmented prompt

        >>> # Grouped mode: accumulate rollouts in replay buffer
        >>> rb = ReplayBuffer(storage=LazyStackStorage(100))
        >>> rb.append_transform(AddFeedbackContext(grpo_size=4))
        >>> # Rollouts are accumulated until grpo_size is reached per prompt
        >>> # Successful rollouts provide feedback for failed ones

    Note:
        When using grouped mode, the transform expects complete trajectories
        (ending with done=True). Incomplete trajectories will raise an error.
    """

    def __init__(
        self,
        grpo_size: int | None = None,
        *,
        prompt_key: NestedKey = "query",
        response_key: NestedKey = ("text", "response"),
        feedback_key: NestedKey = "env_feedback",
        reward_key: NestedKey = ("next", "reward"),
        done_key: NestedKey = ("next", "done"),
        teacher_context_key: NestedKey = "teacher_context",
        success_threshold: float = 0.5,
        include_response_in_context: bool = True,
        template: str | None = None,
        tokenizer: AutoTokenizer | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.grpo_size = grpo_size
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.feedback_key = feedback_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.teacher_context_key = teacher_context_key
        self.success_threshold = success_threshold
        self.include_response_in_context = include_response_in_context
        self.template = template if template is not None else DEFAULT_TEACHER_TEMPLATE
        self.tokenizer = tokenizer
        self.verbose = verbose

        # Storage for grouped mode
        if grpo_size is not None:
            self.queues = defaultdict(lambda: deque(maxlen=grpo_size))
        else:
            self.queues = None

        self.in_keys = [prompt_key, response_key, feedback_key, reward_key, done_key]
        self.out_keys = [teacher_context_key]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Apply transform in forward direction (for env steps)."""
        return self._add_teacher_context(tensordict)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Apply transform in inverse direction (for replay buffer writes).

        In grouped mode, this accumulates rollouts and processes them when
        enough have been collected for each prompt.
        """
        if self.grpo_size is None:
            # Direct mode: just add teacher context
            return self._add_teacher_context(tensordict)

        # Grouped mode: accumulate and process
        return self._process_grouped(tensordict)

    def _add_teacher_context(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Add teacher context to a single tensordict.

        Constructs the feedback-augmented prompt for the self-teacher.
        """
        # Get components
        prompt = self._get_value(tensordict, self.prompt_key, default="")
        response = self._get_value(tensordict, self.response_key, default="")
        env_feedback = self._get_value(tensordict, self.feedback_key, default=None)
        successful_rollout = tensordict.get("_successful_rollout", None)

        # Build feedback section
        if env_feedback is not None and successful_rollout is not None:
            feedback_section = FEEDBACK_WITH_SOLUTION.format(
                successful_rollout=successful_rollout,
                env_output=env_feedback,
            )
        elif env_feedback is not None:
            feedback_section = FEEDBACK_WITHOUT_SOLUTION.format(
                env_output=env_feedback,
            )
        elif successful_rollout is not None:
            feedback_section = SUCCESS_ONLY.format(
                successful_rollout=successful_rollout,
            )
        else:
            # No feedback available - teacher context is just the prompt
            feedback_section = ""

        # Build full teacher context
        if feedback_section:
            teacher_context = self.template.format(
                prompt=prompt,
                feedback_section=feedback_section,
            )
        else:
            teacher_context = prompt

        # Optionally include response
        if self.include_response_in_context and response:
            # The response is appended as what the assistant said
            # (the teacher will re-evaluate these tokens)
            teacher_context_with_response = {
                "prompt": teacher_context,
                "response": response,
            }
            tensordict.set(self.teacher_context_key, teacher_context_with_response)
        else:
            tensordict.set(self.teacher_context_key, teacher_context)

        return tensordict

    def _get_value(self, tensordict: TensorDictBase, key: NestedKey, default=None):
        """Get a value from tensordict, handling nested keys and defaults."""
        try:
            value = tensordict.get(key, default=default)
            if value is None:
                return default
            # Handle various types
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                return value
            return value
        except (KeyError, AttributeError):
            return default

    def _process_grouped(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        """Process tensordict in grouped mode.

        Accumulates rollouts per prompt and processes when grpo_size is reached.
        Successful rollouts provide feedback for failed ones.
        """
        if self.verbose:
            torchrl_logger.info(
                f"Invoking AddFeedbackContext.\nData size: {tensordict.shape}.\n"
                f"Current queue size: {len(self.queues)}.\n"
                f"Total queue content: {sum(len(q) for q in self.queues.values())}"
            )

        # Handle different input dimensions
        if tensordict.ndim == 1:
            # Check how many done states we have
            done = tensordict.get(self.done_key, None)
            if done is None:
                done = torch.ones(1, dtype=torch.bool)  # Assume single complete traj

            num_done = done.sum() if isinstance(done, torch.Tensor) else 1
            if num_done > 1:
                # Split into individual trajectories
                done_idx = done.nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                tensordicts = tensordict.split(splits.tolist())
                tensordicts = [self._process_grouped(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts) if tensordicts else None

            # Single trajectory
            if tensordict.ndim > 0:
                last_done = tensordict[-1].get(self.done_key, None)
            else:
                last_done = tensordict.get(self.done_key, None)

            if last_done is not None and not last_done.all():
                raise RuntimeError("Expected the trajectory to be done.")

            # Get prompt for grouping
            if tensordict.ndim > 0:
                prompt = self._get_value(tensordict[0], self.prompt_key, default="")
            else:
                prompt = self._get_value(tensordict, self.prompt_key, default="")

            if not isinstance(prompt, str):
                # Convert to string for hashing
                prompt = str(prompt)

            self.queues[prompt].append(tensordict)

            if len(self.queues[prompt]) == self.grpo_size:
                if self.verbose:
                    torchrl_logger.info(f"Processing group for {prompt[:50]}...")

                # Process the group
                tds = list(self.queues[prompt])
                del self.queues[prompt]

                # Find successful rollouts
                successful_rollouts = []
                for td in tds:
                    reward = self._get_final_reward(td)
                    if reward is not None and reward > self.success_threshold:
                        response = self._get_value(td, self.response_key, default=None)
                        if response is not None:
                            successful_rollouts.append(response)

                # Add feedback context to each trajectory
                processed_tds = []
                for td in tds:
                    td = td.clone(False)
                    reward = self._get_final_reward(td)
                    is_successful = (
                        reward is not None and reward > self.success_threshold
                    )

                    if not is_successful and successful_rollouts:
                        # Use a successful rollout as feedback
                        td.set("_successful_rollout", successful_rollouts[0])

                    self._add_teacher_context(td)
                    processed_tds.append(td)

                # Stack and return
                return lazy_stack(processed_tds)

            return None  # Not enough rollouts yet

        elif tensordict.ndim > 2:
            # Flatten extra dimensions
            tensordict = tensordict.flatten(0, -2)

        # Process each trajectory
        trajs = tensordict.unbind(0)
        result = []
        for traj in trajs:
            td_out = self._process_grouped(traj)
            if td_out is None:
                continue
            result.append(td_out)

        if result:
            return torch.cat(result, 0)
        return None

    def _get_final_reward(self, tensordict: TensorDictBase) -> float | None:
        """Get the final reward from a trajectory."""
        try:
            if tensordict.ndim > 0:
                reward = tensordict[-1].get(self.reward_key, None)
            else:
                reward = tensordict.get(self.reward_key, None)

            if reward is None:
                return None

            if isinstance(reward, torch.Tensor):
                # Get the final value
                if reward.numel() > 1:
                    reward = reward[-1]
                return float(reward.item())
            return float(reward)
        except (KeyError, AttributeError, TypeError):
            return None

    def reset_queues(self):
        """Reset the accumulation queues (for grouped mode)."""
        if self.queues is not None:
            self.queues.clear()


class BuildTeacherContext(Transform):
    """Builds teacher context from History objects for SDPO.

    This transform is designed for chat-based LLM training where prompts and
    responses are represented as History objects. It constructs the teacher
    context by appending feedback information to the conversation history.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for applying chat templates.

    Keyword Args:
        history_key (NestedKey): Key for the chat history. Defaults to ``"history"``.
        feedback_key (NestedKey): Key for environment feedback. Defaults to ``"env_feedback"``.
        success_response_key (NestedKey): Key for successful response (if available).
            Defaults to ``"successful_response"``.
        teacher_context_key (NestedKey): Key where teacher context will be written.
            Defaults to ``"teacher_context"``.
        chat_template_name (str | None): Name of the chat template to use.
            Defaults to ``None``.

    Example:
        >>> transform = BuildTeacherContext(tokenizer)
        >>> td["history"] = ChatHistory(prompt=prompt_history, response=response_history)
        >>> td["env_feedback"] = "Error: index out of bounds"
        >>> td_out = transform(td)
        >>> # td_out["teacher_context"] now contains the augmented history
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        *,
        history_key: NestedKey = "history",
        feedback_key: NestedKey = "env_feedback",
        success_response_key: NestedKey = "successful_response",
        teacher_context_key: NestedKey = "teacher_context",
        chat_template_name: str | None = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.history_key = history_key
        self.feedback_key = feedback_key
        self.success_response_key = success_response_key
        self.teacher_context_key = teacher_context_key
        self.chat_template_name = chat_template_name

        self.in_keys = [history_key, feedback_key, success_response_key]
        self.out_keys = [teacher_context_key]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Build teacher context from history and feedback."""
        history = tensordict.get(self.history_key, None)
        if history is None:
            return tensordict

        feedback = tensordict.get(self.feedback_key, None)
        success_response = tensordict.get(self.success_response_key, None)

        # Build feedback message
        feedback_parts = []
        if success_response is not None:
            feedback_parts.append(f"Correct solution:\n{success_response}")
        if feedback is not None:
            feedback_parts.append(
                f"The following is feedback from your unsuccessful earlier attempt:\n{feedback}"
            )
        if feedback_parts:
            feedback_parts.append("Correctly solve the original question.")

        feedback_message = "\n\n".join(feedback_parts)

        # Create teacher context by extending history with feedback
        if hasattr(history, "prompt"):
            # ChatHistory-like object
            prompt_history = history.prompt

            if feedback_message:
                # Append feedback as a user message
                feedback_turn = History(
                    role=["user"],
                    content=[feedback_message],
                    batch_size=(1,),
                )
                teacher_prompt = prompt_history.extend(feedback_turn, inplace=False)
            else:
                teacher_prompt = prompt_history

            # Create teacher context dict with prompt and response
            teacher_context = {
                "history": teacher_prompt,
                "original_response": history.response
                if hasattr(history, "response")
                else None,
            }
        else:
            # Plain History object
            if feedback_message:
                feedback_turn = History(
                    role=["user"],
                    content=[feedback_message],
                    batch_size=(1,),
                )
                teacher_context = history.extend(feedback_turn, inplace=False)
            else:
                teacher_context = history

        tensordict.set(self.teacher_context_key, teacher_context)
        return tensordict
