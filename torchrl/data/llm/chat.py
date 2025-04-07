# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re

import torch

from tensordict import lazy_stack, LazyStackedTensorDict, list_to_stack, TensorClass
from tensordict.utils import _maybe_correct_neg_dim
from torchrl._utils import logger as torchrl_logger

_TEMPLATES = {
    "chatml_format": """{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""",
}


class History(TensorClass["nocast"]):
    """A class representing a structured history of messages in a conversation, designed for efficient manipulation and integration with language models.

    The `History` class provides a centralized API for managing conversational data, offering several advantages over
    traditional list-based approaches:

    - Centralized API for conversion to and from string formats, facilitating seamless integration with language models.
    - Efficient methods to append, extend, and reshape history elements, enabling dynamic construction of conversation
      trajectories, especially useful in reinforcement learning environments.
    - Interoperability with the `transformers` API, allowing for easy tokenization and preparation of input data.

    Attributes:
        role (str): The role of the message sender.
        content (str): The content of the message.

    Methods:
        apply_chat_template: converts the `History` object to str / tokens.
        append: append one element to the list of items along a given dimension.
        extend: extend the list of items along a given dimension.

    Examples:
        >>> # With tensordict < 0.10, we need to tell the lib that lists constitute batches
        >>> import tensordict
        >>> tensordict.set_list_to_stack(True).set()
        >>> import transformers
        >>> history0 = History(
        ...     role='system',
        ...     content='''CONTENT
        ... This is the setup''',
        ... )
        >>> history1 = History(
        ...     role='user',
        ...     content='''CONTENT
        ... This is the first user prompt''',
        ... )
        >>> history2 = History(
        ...     role='assistant',
        ...     content='''CONTENT
        ... This is the second prompt, the first for the assistant.''',
        ... )
        >>> history = torch.stack([history0, history1, history2])
        >>> assert history.role == ['system', 'user', 'assistant']
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("GPT2")
        >>> # Apply a template to pass the history to an LLM. Note that the output has
        >>> #  an additional prompt to elict an answer from the LLM thanks to the 'add_generation_prompt' argument.
        >>> parsed_string = history.apply_chat_template(tokenizer=tokenizer, add_generation_prompt=True)
        >>> parsed_string
            <|im_start|>system
        CONTENT
        This is the setup<|im_end|>

            <|im_start|>user
        CONTENT
        This is the first user prompt<|im_end|>

            <|im_start|>assistant
        CONTENT
        This is the second prompt, the first for the assistant.<|im_end|>

        <|im_start|>assistant

    """

    role: str
    content: str

    def __post_init__(self):
        if not list_to_stack():
            raise RuntimeError(
                "Please set the list_to_stack to True using tensordict.set_list_to_stack(True).set() at the beginning of your script, "
                "or the LIST_TO_STACK=1 environment variable."
            )

    def apply_chat_template(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizer,  # noqa
        add_generation_prompt: bool = True,
        chat_template: str = _TEMPLATES["chatml_format"],
        continue_final_message: bool = False,
        tokenize: bool = False,
        padding: bool | str = False,
        truncation: bool | str = False,
        return_tensors: str | None = "pt",
    ):
        """Applies a chat template to the history.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            add_generation_prompt (bool, optional): Whether to add a generation prompt. Defaults to True.
            chat_template (str, optional): The chat template to use. Defaults to _TEMPLATES["chatml_format"].
            continue_final_message (bool, optional): Whether to continue the final message. Defaults to False.
            tokenize (bool, optional): Whether to tokenize the output. Defaults to False.
            padding (bool | str, optional): The padding strategy to use. Defaults to False.
            truncation (bool | str, optional): The truncation strategy to use. Defaults to False.
            return_tensors (str | None, optional): The type of tensors to return. Defaults to "pt".

        Returns:
            The formatted history.
        """
        self_flat = self.view(-1).tolist()
        return tokenizer.apply_chat_template(
            self_flat,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            continue_final_message=continue_final_message,
        )

    @classmethod
    def _inv_chatml(cls, text: str) -> History:
        """Inverts a chatml string into a History object.

        Args:
            text (str): The chatml string to invert.

        Returns:
            History: The inverted History object.
        """
        torchrl_logger.debug(f"Inverting chatml:\n{text}")
        pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        chat_template = []
        for match in matches:
            role = match[0].strip()

            # Override role
            # role = "assistant"
            content = match[1].strip()
            chat_template.append({"role": role, "content": content})
        return cls(
            role=[chat_template["role"] for chat_template in chat_template],
            content=[chat_template["content"] for chat_template in chat_template],
            batch_size=len(chat_template),
        )

    def append(
        self, history: History, *, inplace: bool = True, dim: int = 0
    ) -> History:
        """Appends a new history to the current one.

        Args:
            history (History): The new history to append.
            inplace (bool, optional): Whether to perform the operation in-place. Defaults to True.
            dim (int, optional): The dimension to append along. Defaults to 0.

        Returns:
            History: The appended History object.
        """
        if not self.batch_dims:
            raise RuntimeError(
                "Cannot append an element to a batchless History. Call unsqueeze(dim=0) first on self."
            )
        if self.batch_dims != history.batch_dims + 1:
            raise RuntimeError(
                f"The new history to append must have one less dimension than self. Got self.ndim={self.ndim} and history.ndim={history.ndim}."
            )
        if inplace:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
            if (
                isinstance(self._tensordict, LazyStackedTensorDict)
                and self._tensordict.stack_dim == dim
            ):
                self._tensordict.append(history._tensordict)
                return self
            else:
                td = lazy_stack(
                    list(self._tensordict.unbind(dim)) + [history._tensordict], dim=dim
                )
                self.__dict__["_tensordict"] = td
                return self
        return torch.stack(list(self.unbind(dim)) + [history], dim=dim)

    def extend(
        self, history: History, *, inplace: bool = True, dim: int = 0
    ) -> History:
        if not self.batch_dims:
            raise RuntimeError(
                "Cannot add an element to a batchless History. Call unsqueeze(dim=0) first on self."
            )
        if self.batch_dims != history.batch_dims:
            raise RuntimeError(
                f"The new history to extend must have as many dimensions as self. Got self.ndim={self.ndim} and history.ndim={self.ndim}."
            )
        if inplace:
            dim = _maybe_correct_neg_dim(dim, self.batch_size)
            if (
                isinstance(self._tensordict, LazyStackedTensorDict)
                and self._tensordict.stack_dim == dim
            ):
                self._tensordict.extend(history._tensordict)
                return self
            else:
                td = lazy_stack(
                    list(self._tensordict.unbind(dim))
                    + list(history._tensordict.unbind(dim)),
                    dim=dim,
                )
                self.__dict__["_tensordict"] = td
                return self
        return torch.stack(list(self.unbind(dim)) + list(history.unbind(dim)), dim=dim)
