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
from pprint import pprint

import torch

from tensordict import set_list_to_stack, TensorDict
from torchrl import torchrl_logger
from torchrl.data import CompositeSpec, Unbounded
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
    apply_template=True,
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
    headless=False,  # Set to False to see the browser actions
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

    def _call(self, tensordict: TensorDict) -> TensorDict:
        """Process the tensordict and assign rewards based on the LLM's response.

        Args:
            tensordict (TensorDict): The tensordict containing the environment state.
                Must have a "history" key containing the conversation history.

        Returns:
            TensorDict: The tensordict with an added "reward" key containing the
                computed reward with shape (B, 1) where B is the batch size.
        """
        # ChatEnv has created a history item. We just pick up the last item,
        # and check if `"Paris"` is in the response.
        # We use index 0 because we are in a single-instance environment.
        history = tensordict[0]["history"]
        last_item = history[-1]
        if "Paris" in last_item.content:
            torchrl_logger.info("Found the answer to the question: Paris")
            # Recall that rewards have a trailing singleton dimension.
            tensordict["reward"] = torch.full((1, 1), 2.0)
        # Check if we successfully reached the website
        elif (
            "google.com" in last_item.content
            and "executed successfully" in last_item.content
        ):
            torchrl_logger.info("Reached the website google.com")
            tensordict["reward"] = torch.full((1, 1), 1.0)
        else:
            tensordict["reward"] = torch.full((1, 1), 0.0)
        return tensordict

    def transform_reward_spec(self, reward_spec: CompositeSpec) -> CompositeSpec:
        """Transform the reward spec to include our custom reward.

        This method is required to override the reward spec since the environment
        is initially reward-agnostic.

        Args:
            reward_spec (CompositeSpec): The original reward spec from the environment.

        Returns:
            CompositeSpec: The transformed reward spec with our custom reward definition.
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
# that executes tool actions and displays the results.


def execute_tool_action(
    env: ChatEnv,
    current_state: TensorDict,
    action: str,
    verbose: bool = True,
) -> tuple[TensorDict, TensorDict]:
    """Execute a tool action and show the formatted interaction."""
    s = current_state.set("text_response", [action])
    s, s_ = env.step_and_maybe_reset(s)

    if verbose:
        print("\nLLM Action:")
        print("-----------")
        print(action)
        print("\nEnvironment Response:")
        print("--------------------")
        pprint(s_["history"].apply_chat_template(tokenizer=env.tokenizer))

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
        text=["What is the capital of France?"],
        batch_size=(1,),
    )
)

#####################################################################
# Now we'll navigate to Google using the browser transform. The transform
# expects actions in a specific JSON format wrapped in tool tags.
# In practice, this action should be the output of our LLM which
# will write the response string in the `"text_response"` key.

s, s_ = execute_tool_action(
    env,
    reset,
    """
    Let me search for that:
    <tool>browser
    {
        "action": "navigate",
        "url": "https://google.com"
    }
    </tool><|im_end|>
    """,
)

#####################################################################
# Step 4: Performing the Search
# -----------------------------
#
# With the browser open, we can now type our query and execute the search.
# First, we'll type the search query into Google's search box.

s, s_ = execute_tool_action(
    env,
    s_,
    """
    Let me type the search query:
    <tool>browser
    {
        "action": "type",
        "selector": "[name='q']",
        "text": "What is the capital of France?"
    }
    </tool><|im_end|>
    """,
)

#####################################################################
# Next, we'll click the search button to execute the search. Note how we
# use CSS selectors to identify elements on the page.

s, s_ = execute_tool_action(
    env,
    s_,
    """
    Now let me click the search button:
    <tool>browser
    {
        "action": "click",
        "selector": "[name='btnK']"
    }
    </tool><|im_end|>
    """,
)

#####################################################################
# Step 5: Extracting Results
# --------------------------
#
# Finally, we'll extract the search results from the page. The browser transform
# can extract both text content and HTML from specified elements.

s, s_ = execute_tool_action(
    env,
    s_,
    """
    Let me extract the results:
    <tool>browser
    {
        "action": "extract",
        "selector": "#search",
        "extract_type": "text"
    }
    </tool><|im_end|>
    """,
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
# 1. Understanding TorchRL's LLM environment composition
# 2. Creating and appending tool transforms
# 3. Formatting tool responses and LLM interactions
# 4. Handling tool execution and state management
# 5. Integrating with LLM wrappers (vLLM, Transformers)
#
# See the :ref:`ref_llms` tutorial for more information on how to build tool-enabled
# environments with TorchRL.
