.. currentmodule:: torchrl

LLM interface
=============

.. _ref_llms:

TorchRL offers a set of tools for LLM post-training, as well as some examples for training or setup.

Collectors
----------

TorchRL offers a specialized collector class (:class:`~torchrl.collectors.llm.LLMCollector`) that is tailored for LLM
use cases. We also provide dedicated updaters for some inference engines.

.. currentmodule:: torchrl.collectors.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    vLLMUpdater
    LLMCollector


Data structures
---------------

To handle text-based data structures (such as conversations etc.), we offer a few data structures dedicated to carrying
data for LLM post-training.

.. currentmodule:: torchrl.data.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    History
    ContentBase
    LLMData

Environments
------------

When fine-tuning an LLM using TorchRL, the environment is a crucial component of the inference pipeline, alongside the
policy and collector. Environments manage operations that are not handled by the LLM itself, such as interacting with
tools, loading prompts from datasets, computing rewards (when necessary), and formatting data.

Therefore, the fundamental structure of an LLM post-training pipeline is:

- A policy that wraps the LLM and the LLM only
- An environment that handles the world around the LLM:
    - Loading data (through :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer`)
    - Formatting data (through :class:`~torchrl.envs.llm.transforms.TemplateTransform`)
    - Executing tools (through :class:`~torchrl.envs.llm.transforms.PythonInterpreter`)
    - Computing rewards online, if needed (through :class:`~torchrl.envs.llm.transforms.KLRewardTransform`)
- A data collector that takes the policy (the LLM) and the environment, and handles the inference part of the pipeline:
    - Running reset, step and gathering actions;
    - Yielding the data in a consistent format - or populating a buffer;
    - Updating the policy weights (through :class:`~torchrl.collectors.WeightUpdaterBase` classes)
- A replay buffer that stores the data collected using the collector
- A loss that takes the LLM's output and returns a loss (through :class:`~torchrl.objectives.llm.GRPOLoss` for example)

These elements are presented in the GRPO scripts in the `sota-implementations/llm` directory.

The design of environments in TorchRL allows for flexibility and modularity. By framing tasks as environments, users can
easily extend or modify existing environments using transforms. This approach enables the isolation of individual
components within specific :class:`~torchrl.envs.EnvBase` or :class:`~torchrl.envs.Transform` subclasses, making it
simpler to augment or alter the environment logic.

Available Environment Classes and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchRL provides various environment classes and utilities for working with LLMs, including:

- Various environment classes (:class:`~torchrl.envs.llm.ChatEnv`, :class:`~torchrl.envs.llm.DatasetChatEnv`,
  :class:`~torchrl.envs.llm.GSM8KEnv`, etc.)
- Utility functions (:class:`~torchrl.envs.make_gsm8k_env`, :class:`~torchrl.envs.make_mlgym`, etc.)
- Transforms and other supporting classes (:class:`~torchrl.envs.KLRewardTransform`,
  :class:`~torchrl.envs.TemplateTransform`, :class:`~torchrl.envs.Tokenizer`, etc.)

These components can be used to create customized environments tailored to specific use cases and requirements.

.. currentmodule:: torchrl.envs.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatEnv
    DatasetChatEnv
    GSM8KEnv
    make_gsm8k_env
    GSM8KPrepareQuestion
    IFEvalEnv
    IfEvalScorer
    IFEvalScoreData
    LLMEnv
    LLMHashingEnv
    make_mlgym
    MLGymWrapper
    GSM8KRewardParser

Transforms
~~~~~~~~~~

Transforms are used to modify the data before it is passed to the LLM.
Tools are usually implemented as transforms, and appended to a base environment
such as :class:`~torchrl.envs.llm.ChatEnv`.

An example of a tool transform is the :class:`~torchrl.envs.llm.transforms.PythonInterpreter` transform, which is used
to execute Python code in the context of the LLM.

    >>> from torchrl.envs.llm.transforms import PythonInterpreter
    >>> from torchrl.envs.llm import ChatEnv
    >>> from tensordict import TensorDict, set_list_to_stack
    >>> from transformers import AutoTokenizer
    >>> from pprint import pprint
    >>> set_list_to_stack(True).set()
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    >>> base_env = ChatEnv(
    ...     tokenizer=tokenizer,
    ...     system_prompt="You are an assistant that can execute Python code. Decorate your code with ```python``` tags.",
    ...     user_role="user",
    ...     system_role="system",
    ...     batch_size=[1],
    ... )
    >>> env = base_env.append_transform(PythonInterpreter())
    >>> env.set_seed(0)
    >>> # Pass the reset data - the prompt - to the environment
    >>> reset_data = env.reset(TensorDict(
    ...     text="Let's write a Python function that returns the square of a number.",
    ...     batch_size=[1])
    ... )
    >>> # Simulate an action - i.e., a response from the LLM (as if we were an LLM) 
    >>> action = """Here is a block of code to be executed in python:
    ... ```python
    ... def square(x):
    ...     return x * x
    ... print('testing the square function with input 2:', square(2))
    ... ```
    ... <|im_end|>
    ... """
    >>> step_data = reset_data.set("text_response", [action])
    >>> s, s_ = env.step_and_maybe_reset(reset_data)
    >>> # The history is a stack of chat messages.
    >>> #  The python interpreter transform has executed the code in the last message.
    >>> pprint(s_["history"].apply_chat_template(tokenizer=tokenizer))
    ['<|im_start|>system\n'
     'You are an assistant that can execute Python code. Decorate your code with '
     '```python``` tags.<|im_end|>\n'
     '<|im_start|>user\n'
     "Let's write a Python function that returns the square of a "
     'number.<|im_end|>\n'
     '<|im_start|>assistant\n'
     'Here is a block of code to be executed in python:\n'
     '```python\n'
     'def square(x):\n'
     '    return x * x\n'
     "print('testing the square function with input 2:', square(2))\n"
     '```<|im_end|>\n'
     '<|im_start|>user\n'
     '<tool_response>\n'
     'Code block 1 executed successfully:\n'
     'testing the square function with input 2: 4\n'
     '\n'
     '</tool_response><|im_end|>\n'
     '<|im_start|>assistant\n']

Similarly, environments that load data from a dataset are just special instances of the :class:`~torchrl.envs.llm.ChatEnv`
augmented with a :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer` transforms (and some dedicated reward parsing
transforms).

.. currentmodule:: torchrl.envs.llm.transforms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DataLoadingPrimer
    KLRewardTransform
    PythonInterpreter
    TemplateTransform
    Tokenizer
    as_nested_tensor
    as_padded_tensor

Modules
-------

The :ref:`~torchrl.modules.llm` section provides a set of wrappers and utility functions for popular training and
inference backends. The main goal of these primitives is to:

- Unify the input / output data format across training and inference pipelines;
- Unify the input / output data format across backends (to be able to use different backends across losses and
  collectors, for instance)
- Give appropriate tooling to construct these objects in typical RL settings (resource allocation, async execution,
  weight update, etc.)

Wrappers
~~~~~~~~

.. currentmodule:: torchrl.modules.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TransformersWrapper
    vLLMWrapper

Utils
~~~~~

.. currentmodule:: torchrl.modules.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    CategoricalSequential
    LLMOnDevice
    make_vllm_worker
    stateless_init_process_group
    vLLMWorker

Objectives
----------

LLM post training require some appropriate versions of the losses implemented in TorchRL.

GRPO
~~~~

.. currentmodule:: torchrl.objectives.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    GRPOLoss
    GRPOLossOutput
    MCAdvantage
