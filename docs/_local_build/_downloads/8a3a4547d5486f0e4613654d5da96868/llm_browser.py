"""
TorchRL LLM: Building Tool-Enabled Environments
===============================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _llm_tools:

This tutorial demonstrates how to build and compose LLM environments with tool capabilities
in TorchRL. We'll show how to create a complete environment that can execute tools,
format responses, and handle interactions between the LLM and external tools.

The tutorial uses web browsing as a concrete example, but the concepts apply to any
tool integration in TorchRL's LLM framework.

Main takeaways:

- Understanding TorchRL's LLM environment composition
- Creating and appending tool transforms
- Formatting tool responses and LLM interactions
- Handling tool execution and state management

Prerequisites: Basic familiarity with TorchRL's environment concepts.
"""

#####################################################################
# Installation
# ------------
#
# First, install TorchRL with LLM support. If you're running this in a Jupyter
# notebook, you can install the packages using:
#
# .. code-block:: bash
#
#     %pip install "torchrl[llm]"    # Install TorchRL with all LLM dependencies
#
# The `torchrl[llm]` package includes all necessary dependencies for LLM functionality,
# including transformers, vllm, and playwright for browser automation.
#
# After installation, you'll need to set up the browser automation components:
#
# .. code-block:: bash
#
#     !playwright install            # Install browser binaries
#
# Note: The `!` and `%pip` prefixes are specific to Jupyter notebooks. In a regular
# terminal, use these commands without the prefixes.

#####################################################################
# Environment Setup
# -----------------
#
# TorchRL's LLM interface is built around composable environments and transforms.
# The key components are:
#
# 1. A base environment (ChatEnv)
# 2. Tool execution transforms
# 3. Data loading transforms
# 4. Reward computation transforms
#
# Let's import the necessary components and set up our environment.

from __future__ import annotations

import warnings

import torch

from tensordict import lazy_stack, set_list_to_stack, TensorDict
from torchrl import logger as torchrl_logger
from torchrl.data import Unbounded
from torchrl.data.llm import History
from torchrl.envs import Transform
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms.browser import BrowserTransform
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

#####################################################################
# Step 1: Basic Environment Configuration
# ---------------------------------------
#
# We'll create a ChatEnv and configure it with browser automation capabilities.
# First, we enable list-to-stack conversion for TensorDict, which is required
# for proper batch handling in LLM environments.

# Enable list-to-stack conversion for TensorDict
set_list_to_stack(True).set()

#####################################################################
# Now we'll create the tokenizer and base environment. The environment requires
# a batch size, even if we're only running a single instance.

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
env = ChatEnv(
    batch_size=(1,),
    tokenizer=tokenizer,
    system_prompt=(
        "You are a helpful assistant that can use tools to accomplish tasks. "
        "Tools will be executed and their responses will be added to our conversation."
    ),
)

#####################################################################
# Next, we'll add the browser transform with safety configurations. This transform
# enables web browsing capabilities with domain restrictions for security.

browser_transform = BrowserTransform(
    allowed_domains=["google.com", "github.com"],
    headless=True,  # Set to False to see the browser actions
)
env = env.append_transform(browser_transform)

#####################################################################
# We can also design a transform to assign rewards to the environment.
# For example, we can parse the result of the browser transform to assign a reward
# whenever specific goals are achieved. Very simply, in this example, we will assign
# a reward of 2 if the LLM finds the answer to the question (Paris), a reward of 1 if it
# reaches the desired website, and a reward of 0 otherwise.


class RewardTransform(Transform):
    """A transform that assigns rewards based on the LLM's responses.

    This transform parses the browser responses in the environment's history and assigns
    rewards based on specific achievements:

    - Finding the correct answer (Paris): reward = 2.0
    - Successfully reaching Google: reward = 1.0
    - Otherwise: reward = 0.0

    """

    def _step(
        self, tensordict: TensorDict, next_tensordict: TensorDict
    ) -> TensorDict:
        """Process the tensordict and assign rewards based on the LLM's response.

        Args:
            tensordict (TensorDict): The input tensordict.
            next_tensordict (TensorDict): The next tensordict containing the environment state.
                Must have a "history" key containing the conversation history.

        Returns:
            TensorDict: The tensordict with an added "reward" key containing the
                computed reward with shape (B, 1) where B is the batch size.
        """
        # ChatEnv has created a history item. We just pick up the last item,
        # and check if `"Paris"` is in the response.
        # We use index 0 because we are in a single-instance environment.
        history = next_tensordict[0]["history"].prompt
        last_item = history[-1]
        if "Paris" in last_item.content:
            torchrl_logger.info("Found the answer to the question: Paris")
            # Recall that rewards have a trailing singleton dimension.
            next_tensordict["reward"] = torch.full((1, 1), 2.0)
        # Check if we successfully reached the website
        elif "google.com" in last_item.content and "success" in last_item.content.lower():
            torchrl_logger.info("Reached the website google.com")
            next_tensordict["reward"] = torch.full((1, 1), 1.0)
        else:
            next_tensordict["reward"] = torch.full((1, 1), 0.0)
        return next_tensordict

    def transform_reward_spec(self, reward_spec):
        """Transform the reward spec to include our custom reward.

        This method is required to override the reward spec since the environment
        is initially reward-agnostic.

        Args:
            reward_spec: The original reward spec from the environment.

        Returns:
            The transformed reward spec with our custom reward definition.
            The reward will have shape (B, 1) where B is the batch size.
        """
        reward_spec["reward"] = Unbounded(
            shape=reward_spec.shape + (1,), dtype=torch.float32
        )
        return reward_spec


# We append the reward transform to the environment.
env = env.append_transform(RewardTransform())

#####################################################################
# Step 2: Tool Execution Helper
# -----------------------------
#
# To make our interaction with tools more organized, we'll create a helper function
# that simulates LLM responses and executes tool actions.
#
# In a real scenario, this would be handled by an LLM policy (like
# :class:`~torchrl.modules.llm.policies.TransformersWrapper` or
# :class:`~torchrl.modules.llm.policies.vLLMWrapper`). Here we simulate the
# LLM's output manually to demonstrate the environment flow.
#
# The key insight is that the ChatEnv works with a :class:`~torchrl.data.llm.History`
# object that tracks the conversation. The LLM policy:
#
# 1. Receives `history.prompt` (the conversation so far)
# 2. Generates a response
# 3. Sets `history.full` to the complete conversation (prompt + response)
# 4. Sets `history.response` to just the new response


def simulate_llm_response(
    env: ChatEnv,
    current_state: TensorDict,
    action: str,
    verbose: bool = True,
) -> tuple[TensorDict, TensorDict]:
    """Simulate an LLM response and step the environment.

    This function mimics what an LLM policy would do: take the current
    conversation state, generate a response, and return the updated state.

    Args:
        env: The chat environment.
        current_state: Current tensordict with the conversation history.
        action: The simulated LLM response text.
        verbose: Whether to print the interaction.

    Returns:
        tuple: (s, s_) where s contains the full step result and s_ is the
            next state (used as input to the next step).
    """
    # Create an assistant response as a History object
    assistant_response = History(role="assistant", content=action)
    # Add batch dimensions to match the environment's batch size
    assistant_response = lazy_stack([assistant_response]).unsqueeze(-1)

    # Get the current prompt and extend it with the response
    prompt = current_state["history"].prompt
    full = prompt.extend(assistant_response, dim=-1)

    # Set the full history and response on the ChatHistory object
    current_state["history"].full = full
    current_state["history"].response = assistant_response

    # Step the environment
    s, s_ = env.step_and_maybe_reset(current_state)

    if verbose:
        print("\nLLM Action:")
        print("-----------")
        print(action)
        print("\nEnvironment Response:")
        print("--------------------")
        # Print the formatted history with the tool response
        torchrl_logger.info(s_["history"].prompt.apply_chat_template(tokenizer=env.tokenizer))
        print(f"Reward: {s['next', 'reward'].item()}")

    return s, s_


#####################################################################
# Step 3: Starting the Interaction
# --------------------------------
#
# Let's begin by initializing the environment with a question and navigating
# to a search engine. Note that the tensordict used as input to the environment
# must share the same batch size as the environment. The text query is put in a list
# of length 1, such that it is compatible with the environment's batch size.

reset = env.reset(
    TensorDict(
        query=["What is the capital of France?"],
        batch_size=(1,),
    )
)

#####################################################################
# Now we'll navigate to Google using the browser transform. The transform
# expects actions in a specific format: ``<tool>tool_name\n{json_args}\n</tool>``.
# In practice, this action would be the output of our LLM policy.

s, s_ = simulate_llm_response(
    env,
    reset,
    """Let me search for that:
<tool>browser
{"action": "navigate", "url": "https://google.com"}
</tool>""",
)

#####################################################################
# Step 4: Performing the Search
# -----------------------------
#
# With the browser open, we can now type our query and execute the search.
# First, we'll type the search query into Google's search box.

s, s_ = simulate_llm_response(
    env,
    s_,
    """Let me type the search query:
<tool>browser
{"action": "type", "selector": "[name='q']", "text": "What is the capital of France?"}
</tool>""",
)

#####################################################################
# Next, we'll click the search button to execute the search. Note how we
# use CSS selectors to identify elements on the page.

s, s_ = simulate_llm_response(
    env,
    s_,
    """Now let me click the search button:
<tool>browser
{"action": "click", "selector": "[name='btnK']"}
</tool>""",
)

#####################################################################
# Step 5: Providing the Answer
# ----------------------------
#
# Finally, the LLM provides the answer based on the search results.
# This triggers the reward transform to assign a reward of 2.0 since
# the answer contains "Paris".

s, s_ = simulate_llm_response(
    env,
    s_,
    """Based on the search results, the capital of France is Paris.""",
)

#####################################################################
# Let's close the environment.
env.close()

#####################################################################
# Conclusion
# ----------
#
# This tutorial demonstrates how to build and compose LLM environments with tool capabilities
# in TorchRL. We've shown how to create a complete environment that can execute tools,
# format responses, and handle interactions between the LLM and external tools.
#
# The key concepts are:
#
# 1. **ChatEnv**: The base environment that manages conversation state using
#    :class:`~torchrl.data.llm.History` objects
# 2. **Tool Transforms**: Like :class:`~torchrl.envs.llm.BrowserTransform` that
#    execute tools and inject results back into the conversation
# 3. **Reward Transforms**: Custom transforms that assign rewards based on
#    conversation content
# 4. **History Management**: Understanding how `history.prompt`, `history.full`,
#    and `history.response` work together in the step cycle
#
# In a real application, you would replace the `simulate_llm_response` helper
# with an actual LLM policy. See the :ref:`ref_llms` documentation for examples
# of using :class:`~torchrl.modules.llm.policies.TransformersWrapper` and
# :class:`~torchrl.modules.llm.policies.vLLMWrapper` with these environments.
