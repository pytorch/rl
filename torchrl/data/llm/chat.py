# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import dataclasses

import re
from typing import Literal

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
        **kwargs,
    ):
        """Applies a chat template to the history.

        Keyword Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            add_generation_prompt (bool, optional): Whether to add a generation prompt. Defaults to True.
            chat_template (str, optional): The chat template to use. Defaults to _TEMPLATES["chatml_format"].
            continue_final_message (bool, optional): Whether to continue the final message. Defaults to False.
            tokenize (bool, optional): Whether to tokenize the output. Defaults to False.
            padding (bool | str, optional): The padding strategy to use. Defaults to False.
            truncation (bool | str, optional): The truncation strategy to use. Defaults to False.
            return_tensors (str | None, optional): The type of tensors to return. Defaults to "pt".
            **kwargs: Additional keyword arguments to pass to the tokenizer `apply_chat_template` method.

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
    def inv_chat_template(
        cls, text: str, chat_template_name: Literal["chatml_format"] = "chatml_format"
    ) -> History:
        if chat_template_name not in ("chatml_format",):
            # Hard coded for now
            raise NotImplementedError(
                "chat_template_name must be one of ('chatml_format',)"
            )
        if isinstance(text, list):
            return torch.stack([cls._inv_chatml(text) for text in text])
        return cls._inv_chatml(text)

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
        roles = []
        contents = []
        for match in matches:
            role = match[0].strip()

            # Override role
            # role = "assistant"
            content = match[1].strip()
            roles.append(role)
            contents.append(content)
        if not roles:
            raise RuntimeError(
                f"Couldn't get a single item out of text {text}. A common cause "
                f"if that special tokens should not be ommitted, did you set include_stop_str_in_output/skip_special_tokens=False?"
            )

        return cls(
            role=roles,
            content=contents,
            batch_size=len(roles),
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
                td = history._tensordict
                if td.device != self.device:
                    if self.device is None:
                        td = td.copy().clear_device_()
                    else:
                        td = td.to(self.device)
                self._tensordict.append(td)
                return self
            else:
                td = history._tensordict
                if td.device != self.device:
                    if self.device is None:
                        td = td.copy().clear_device_()
                    else:
                        td = td.to(self.device)
                td = lazy_stack(list(self._tensordict.unbind(dim)) + [td], dim=dim)
                self.__dict__["_tensordict"] = td
                return self
        if history.device != self.device:
            if self.device is None:
                history = history.copy().clear_device_()
            else:
                history = history.to(self.device)
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
                td = history._tensordict
                if td.device != self.device:
                    if self.device is None:
                        td = td.copy().clear_device_()
                    else:
                        td = td.to(self.device)
                self._tensordict.extend(td)
                return self
            else:
                td = lazy_stack(
                    list(self._tensordict.unbind(dim))
                    + list(history._tensordict.unbind(dim)),
                    dim=dim,
                )
                if td.device != self.device:
                    if self.device is None:
                        td = td.copy().clear_device_()
                    else:
                        td = td.to(self.device)
                self.__dict__["_tensordict"] = td
                return self
        if history.device != self.device:
            if self.device is None:
                history = history.copy().clear_device_()
            else:
                history = history.to(self.device)
        return torch.stack(list(self.unbind(dim)) + list(history.unbind(dim)), dim=dim)

    @classmethod
    def default_spec(cls, shape=(-1,)):
        """A default spec to use in transforms / envs that return History objects.

        Args:
            shape (torch.Size, optional): The shape of the returned History spec. Defaults to `(-1)` (variable length
                along time dimension).

        Example:
            >>> import tensordict
            >>> from torchrl.data import History
            >>> tensordict.set_list_to_stack(True).set()
            >>>
            >>> history = History(role=["system", "user"], content=["a message", "another message"], batch_size=(2,))
            >>> spec = history.default_spec()
            >>> print(spec)
            Composite(
                role: NonTensor(
                    shape=torch.Size([-1]),
                    space=None,
                    device=None,
                    dtype=None,
                    domain=None,
                    example_data=foo),
                content: NonTensor(
                    shape=torch.Size([-1]),
                    space=None,
                    device=None,
                    dtype=None,
                    domain=None,
                    example_data=foo),
                device=None,
                shape=torch.Size([-1]))
            >>> print(spec.zero())
            History(
                content=NonTensorData(data=foo, batch_size=torch.Size([1]), device=None),
                role=NonTensorData(data=foo, batch_size=torch.Size([1]), device=None),
                batch_size=torch.Size([1]),
                device=None,
                is_shared=False)

        """
        from torchrl.data import Composite, NonTensor

        def get_default_value(field):
            if field.default is not dataclasses.MISSING:
                return field.default
            elif field.type in (str, "str"):
                return "foo"
            else:
                return None

        defaults = {
            k: NonTensor(
                example_data=get_default_value(cls.__dataclass_fields__[k]), shape=(-1,)
            )
            for k in cls.__dataclass_fields__
        }

        return Composite(defaults, shape=shape, data_cls=cls)
