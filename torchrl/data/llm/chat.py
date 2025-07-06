# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import dataclasses

import re
from typing import Literal

import torch

from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    list_to_stack,
    TensorClass,
    TensorDict,
)
from tensordict.utils import _maybe_correct_neg_dim
from torchrl._utils import logger as torchrl_logger


_CHAT_TEMPLATES = {
    "chatml_format": """{% for message in messages %}
    {%- if message['role'] == 'assistant' %}
    {% generation %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endgeneration %}
    {%- else %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {%- endif %}
{% endfor %}
{%- if add_generation_prompt %}
    {% generation %}{{- '<|im_start|>assistant\n' }}{% endgeneration %}
{%- endif %}
""",
    "qwen": """
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
    {% generation %}    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}    {% endgeneration %}
    {%- elif message.role == "assistant" %}
        {% generation %}{{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\\\"name\\\": \\\"' }}
            {{- tool_call.name }}
            {{- '\\\", \\\"arguments\\\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {% generation %}{{- '<|im_start|>assistant\\n' }}{% endgeneration %}
{%- endif %}
""",
}


# We need the 'shadow' flag to avoid having tensordict complaining about 'type'/'size' etc. fields
class ContentBase(TensorClass["nocast", "shadow"]):
    """Base class for all message content types.

    Attributes:
        type (str): The type of the content.
        text (str, optional): The text content.
        url (str, optional): The URL content.
        data (str, optional): The data content.
        mime_type (str, optional): The MIME type of the content.
        name (str, optional): The name of the content.
        size (int, optional): The size of the content.
        function_name (str, optional): The name of the function.
        function_args (dict, optional): The arguments of the function.

    Examples:
        >>> from tensordict import lazy_stack
        >>> content1 = ContentBase(type="text", text="Hello, world!")
        >>> print(content1)
        ContentBase(
            text=NonTensorData(data=Hello, world!, batch_size=torch.Size([]), device=None),
            type=NonTensorData(data=text, batch_size=torch.Size([]), device=None),
            url=None,
            data=None,
            mime_type=None,
            name=None,
            size=None,
            function_name=None,
            function_args=None,
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> content2 = ContentBase(type="image", url="https://example.com/image.jpg")
        >>> print(content2)
        ContentBase(
            type=NonTensorData(data=image, batch_size=torch.Size([]), device=None),
            url=NonTensorData(data=https://example.com/image.jpg, batch_size=torch.Size([]), device=None),
            text=None,
            data=None,
            mime_type=None,
            name=None,
            size=None,
            function_name=None,
            function_args=None,
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> content = lazy_stack([content1, content2])
        >>> print(content)
        ContentBase(
            type=NonTensorStack(
                ['text', 'image'],
                batch_size=torch.Size([2]),
                device=None),
            url=None,
            data=None,
            mime_type=None,
            name=None,
            size=None,
            function_name=None,
            function_args=None,
            text=None,
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)
        >>> # A content is typically used in a History object. Usually, its batch dimension is
        >>> #  one dimension greater than the History object.
        >>> history = History(role="user", content=content)

    """

    type: Literal[
        "text", "image", "audio", "video", "file", "function_call"
    ]  # Required: "text", "image", "audio", "video", "file", "function_call"

    # Text content
    text: str | None = None

    # Media/file content (either URL or data)
    url: str | None = None  # HTTP URL to content
    data: str | None = None  # Base64 encoded content

    # Metadata
    mime_type: str | None = None  # "image/jpeg", "audio/mp3", "application/pdf"
    name: str | None = None  # Original filename or description
    size: int | None = None  # File size in bytes

    # Function calling (for AI agents)
    function_name: str | None = None
    function_args: dict | None = None


class History(TensorClass["nocast"]):
    """A class representing a structured history of messages in a conversation, designed for efficient manipulation and integration with language models.

    The `History` class provides a centralized API for managing conversational data, offering several advantages over
    traditional list-based approaches:

    - Centralized API for conversion to and from string formats, facilitating seamless integration with language models.
    - Efficient methods to append, extend, and reshape history elements, enabling dynamic construction of conversation
      trajectories, especially useful in reinforcement learning environments.
    - Interoperability with the `transformers` API, allowing for easy tokenization and preparation of input data.

    .. note:: The `"<none>"` role is used to indicate that the element is a placeholder,
        for example when the tool call was not executed but a stack requires a certain number of elements
        per batch to have congruent shapes. The :meth:`~torchrl.data.llm.chat.History.apply_chat_template`
        method will remove the `<none>` role from the history.

    Attributes:
        role (str): The role of the message sender.
        content (str): The content of the message.
        is_complete (bool): Whether the message was properly terminated with an end token. Defaults to `True`.
        tool_calls (list[dict] | None): Optional list of tool calls in the message.
        tool_responses (list[str] | None): Optional list of tool responses.

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
    content: str | ContentBase
    is_complete: bool = True
    tool_calls: list[dict] | None = None
    tool_responses: list[str] | None = None

    def __post_init__(self):
        if not list_to_stack():
            raise RuntimeError(
                "Please set the list_to_stack to True using tensordict.set_list_to_stack(True).set() at the beginning of your script, "
                "or the LIST_TO_STACK=1 environment variable."
            )

    def apply_chat_template(
        self,
        *,
        tokenizer: transformers.AutoTokenizer | transformers.AutoProcessor,  # noqa
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        chat_template_name: Literal["chatml_format", "qwen"] | None = None,
        continue_final_message: bool = False,
        tokenize: bool | None = None,
        padding: bool | str = False,
        truncation: bool | str = False,
        return_tensors: str | None = None,
        return_dict: bool | None = None,
        return_assistant_tokens_mask: bool = False,
        **kwargs,
    ):
        """Applies a chat template to the history.

        Keyword Args:
            tokenizer (transformers.PreTrainedTokenizer | transformers.AutoProcessor): The tokenizer to use.
            add_generation_prompt (bool, optional): Whether to add a generation prompt (e.g. `"<|im_start|>assistant"`). Defaults to `True`.
            chat_template (str, optional): The chat template to use. Defaults to the tokenizer's default template.
            chat_template_name (Literal["chatml_format", "qwen"], optional): The name of the chat template to use.
                Prevalent over `tokenizer.chat_template`. Defaults to `None`.
            continue_final_message (bool, optional): Whether to continue the final message. Defaults to `False`.
            tokenize (bool, optional): Whether to tokenize the output. Defaults to `False`.
            padding (bool | str, optional): The padding strategy to use. Defaults to `False`.
            truncation (bool | str, optional): The truncation strategy to use. Defaults to `False`.
            return_tensors (str | None, optional): The type of tensors to return. Defaults to "pt".
            return_dict (bool, optional): Whether to return a dictionary. Defaults to `False`.
            return_assistant_tokens_mask (bool, optional): Whether to return a mask of the assistant generated tokens.
                If `True`, the mask will be written to the `assistant_masks` key.
                For tokens generated by the assistant, the mask will contain `1`.
                For user and system tokens, the mask will contain `0`.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
                Defaults to `False`.

                .. note:: By default, the `"qwen"` chat template does not support this functionality. A modified version of the template
                    can be used by setting `chat_template_name="qwen"`, which will override the default template from the tokenizer.
                    For other tokenizers, similar edits can be made to the template and passed to the method via the `chat_template` argument.

            **kwargs: Additional keyword arguments to pass to the tokenizer `apply_chat_template` method.

        Returns:
            The formatted history.
        """
        if chat_template is None:
            if chat_template_name is not None:
                chat_template = _CHAT_TEMPLATES[chat_template_name]
                chat_template_name = None
            elif tokenizer is None:
                raise RuntimeError(
                    "You must specify a tokenizer to use when chat_template is not specified."
                )
            elif "qwen" in getattr(tokenizer, "name_or_path", "").lower():
                # We prefer our implementation of the Qwen template,
                #  since it accounts for the assistant's masking.
                chat_template = _CHAT_TEMPLATES["qwen"]
                chat_template_name = None
            else:
                chat_template = tokenizer.chat_template
        if chat_template is None:
            chat_template = _CHAT_TEMPLATES["chatml_format"]
        if tokenize is None:
            if return_assistant_tokens_mask or return_tensors is not None:
                tokenize = True
            else:
                tokenize = False
        if tokenize:
            if return_tensors is None:
                return_tensors = "pt"
            if return_dict is None and return_assistant_tokens_mask:
                return_dict = True
        elif return_dict is None:
            return_dict = False

        if self.ndim > 1:
            result = [
                self[i].apply_chat_template(
                    tokenizer=tokenizer,
                    add_generation_prompt=add_generation_prompt,
                    chat_template=chat_template,
                    chat_template_name=chat_template_name,
                    tokenize=tokenize,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                    continue_final_message=continue_final_message,
                    return_dict=return_dict,
                    return_assistant_tokens_mask=return_assistant_tokens_mask,
                    **kwargs,
                )
                for i in range(self.batch_size[0])
            ]
            if return_dict:
                return lazy_stack(result)
            else:
                return result
        self_flat = self.view(-1)
        # tolist_first=True is needed to avoid having a list of dict of dicts, but a list of dicts of lists of dicts
        self_flat = self_flat.tolist(tolist_first=True)
        # Remove the "<none>" role
        self_flat = [item for item in self_flat if item["role"] != "<none>"]
        result = tokenizer.apply_chat_template(
            conversation=self_flat,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            continue_final_message=continue_final_message,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            **kwargs,
        )
        if not isinstance(result, (torch.Tensor, list, str)):
            result = TensorDict.from_dict(result, auto_batch_size=True, batch_dims=1)
            # If self has a batch_dims of 1, we have just the time dimension, so we need to remove the batch dim from the result
            if self.batch_dims == 1:
                if result.batch_size[0] != 1:
                    raise RuntimeError(
                        f"Expected a batch size of 1, got {result.batch_size[0]}."
                    )
                result = result.squeeze(0)
        return result

    @classmethod
    def from_text(
        cls,
        text: str | list[str],
        chat_template_name: Literal["chatml_format", "qwen"] | None = None,
        chat_template: str | None = None,
        tokenizer: transformers.AutoTokenizer  # noqa: F821
        | transformers.AutoProcessor  # noqa: F821
        | None = None,
    ) -> History:
        if chat_template_name is None and chat_template is None:
            if "qwen" in getattr(tokenizer, "name_or_path", "").lower():
                # We can automatically detect the template name from the tokenizer
                #  and use the precoded parser.
                chat_template_name = "qwen"
            else:
                chat_template_name = "chatml_format"
        elif chat_template_name in ("chatml_format",):
            func = cls._inv_chatml
        elif chat_template_name in ("qwen",):
            func = cls._inv_qwen
        else:
            raise NotImplementedError(
                "chat_template_name must be one of ('chatml_format', 'qwen')"
            )
        if isinstance(text, list):
            list_of_histories = [func(t) for t in text]
            try:
                return lazy_stack(list_of_histories)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to stack histories: {list_of_histories=}"
                ) from e
        return func(text)

    @classmethod
    def _inv_chatml(cls, text: str) -> History:
        """Inverts a chatml string into a History object.

        Args:
            text (str): The chatml string to invert.

        Returns:
            History: The inverted History object.
        """
        import json

        torchrl_logger.debug(f"Inverting chatml:\n{text}")
        # Find all complete blocks (ending with im_end or endoftext)
        complete_pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|(im_end|endoftext)\|>"
        complete_matches = re.findall(complete_pattern, text, flags=re.DOTALL)

        # Find any incomplete block at the end
        incomplete_pattern = r"<\|im_start\|>(.*?)\n(.*?)$"
        incomplete_matches = []
        if complete_matches:
            # Look for incomplete block after the last complete one
            last_complete = complete_matches[-1]
            last_complete_text = f"<|im_start|>{last_complete[0]}\n{last_complete[1]}<|{last_complete[2]}|>"
            remaining_text = text[
                text.rindex(last_complete_text) + len(last_complete_text) :
            ]
            if remaining_text.strip():
                incomplete_match = re.search(
                    incomplete_pattern, remaining_text, flags=re.DOTALL
                )
                if incomplete_match:
                    incomplete_matches = [
                        (incomplete_match.group(1), incomplete_match.group(2), None)
                    ]
        else:
            # No complete blocks, check entire text for incomplete block
            incomplete_match = re.search(incomplete_pattern, text, flags=re.DOTALL)
            if incomplete_match:
                incomplete_matches = [
                    (incomplete_match.group(1), incomplete_match.group(2), None)
                ]

        # Combine complete and incomplete matches
        matches = complete_matches + incomplete_matches

        # Define tool patterns - same as Qwen for consistency
        tool_call_pattern = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
        tool_response_pattern = re.compile(
            r"<tool_response>\n(.*?)\n</tool_response>", re.DOTALL
        )

        parsed_messages = []
        for match in matches:
            role = match[0].strip()
            content = match[1].strip()
            is_complete = match[2] is not None  # None indicates incomplete

            # Initialize message dict
            message_dict = {
                "role": role,
                "content": content,
                "is_complete": is_complete,
                "tool_calls": None,
                "tool_responses": None,
            }

            # Find tool calls within the message
            tool_calls = tool_call_pattern.findall(content)
            if tool_calls:
                tool_calls_list = []
                for tool_call in tool_calls:
                    try:
                        tool_call_dict = json.loads(tool_call)
                        tool_calls_list.append(tool_call_dict)
                    except json.JSONDecodeError:
                        continue
                if tool_calls_list:
                    message_dict["tool_calls"] = tool_calls_list

            # Check for tool responses
            tool_responses = tool_response_pattern.findall(content)
            if tool_responses:
                message_dict["tool_responses"] = tool_responses

            parsed_messages.append(cls(**message_dict))

        if not parsed_messages:
            raise RuntimeError(
                f"Couldn't get a single item out of text {text}. A common cause "
                f"if that special tokens should not be ommitted, did you set include_stop_str_in_output/skip_special_tokens=False?"
            )

        return lazy_stack(parsed_messages)

    @classmethod
    def _inv_qwen(cls, template):
        import json

        # Define regex patterns for different parts of the template
        message_pattern = re.compile(
            r"<\|im_start\|>(.*?)(?:<\|(im_end|endoftext)\|>|$)", re.DOTALL
        )
        tool_call_pattern = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
        tool_response_pattern = re.compile(
            r"<tool_response>\n(.*?)\n</tool_response>", re.DOTALL
        )

        # Find all messages and track if they end with a proper token
        messages = []
        is_complete_list = []
        for match in message_pattern.finditer(template):
            full_match = match.group(0)
            messages.append(match.group(1))
            # Check if the message ends with a proper token
            is_complete_list.append(
                full_match.endswith("<|im_end|>")
                or full_match.endswith("<|endoftext|>")
            )

        parsed_messages = []
        for message, is_complete in zip(messages, is_complete_list):
            # Split the message into role and content
            parts = message.split("\n", 1)
            if len(parts) < 2:
                continue
            role, content = parts[0], parts[1]

            # Initialize message dict
            message_dict = {
                "role": role.strip(),
                "content": content.strip(),
                "is_complete": is_complete,
                "tool_calls": None,
                "tool_responses": None,
            }

            # Find tool calls within the message
            tool_calls = tool_call_pattern.findall(content)
            if tool_calls:
                tool_calls_list = []
                for tool_call in tool_calls:
                    try:
                        tool_call_dict = json.loads(tool_call)
                        tool_calls_list.append(tool_call_dict)
                    except json.JSONDecodeError:
                        continue
                if tool_calls_list:
                    message_dict["tool_calls"] = tool_calls_list

            # Check for tool responses
            tool_responses = tool_response_pattern.findall(content)
            if tool_responses:
                message_dict["tool_responses"] = tool_responses

            parsed_messages.append(cls(**message_dict))

        if not parsed_messages:
            raise RuntimeError(
                f"Couldn't get a single item out of text {template}. A common cause "
                f"if that special tokens should not be ommitted, did you set include_stop_str_in_output/skip_special_tokens=False?"
            )

        return lazy_stack(parsed_messages)

    def append(
        self, history: History, *, inplace: bool = True, dim: int = -1
    ) -> History:
        """Appends a new history to the current one.

        Args:
            history (History): The new history to append.
            inplace (bool, optional): Whether to perform the operation in-place. Defaults to `True`.
            dim (int, optional): The dimension to append along. Defaults to -1.

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
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        # if self.ndim > 1 and dim >= self.ndim - 1:
        #     # then we need to append each element independently
        #     result = []
        #     for hist, new_hist in zip(self.unbind(0), history.unbind(0)):
        #         hist_c = hist.append(new_hist, inplace=inplace, dim=dim - 1)
        #         result.append(hist_c)
        #     if inplace:
        #         return self
        #     return lazy_stack(result)
        if inplace:
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
        return lazy_stack(list(self.unbind(dim)) + [history], dim=dim)

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
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        # if self.ndim > 1 and dim >= self.ndim - 1:
        #     # then we need to append each element independently
        #     result = []
        #     for hist, new_hist in zip(self.unbind(0), history.unbind(0)):
        #         hist_c = hist.extend(new_hist, inplace=inplace, dim=dim - 1)
        #         result.append(hist_c)
        #     if inplace:
        #         return self
        #     return lazy_stack(result)
        if inplace:
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
                along the time dimension).

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
                example_data=get_default_value(cls.__dataclass_fields__[k]),
                shape=shape,
            )
            for k in cls.__dataclass_fields__
        }

        return Composite(defaults, shape=shape[:-1], data_cls=cls)

    @classmethod
    def from_chats(cls, chats: list[list[dict]]) -> History:
        """Create a History object from a list of chats.

        Args:
            chats (list[list[dict]]): A list of chats, where each chat is a list of dictionaries.
        """
        if isinstance(chats[0], dict):
            return lazy_stack([cls(**chat) for chat in chats])
        else:
            return lazy_stack([cls.from_chats(chat) for chat in chats])
