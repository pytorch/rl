# add_chat_template

*class*torchrl.data.llm.add_chat_template(*template_name: str*, *template: str*, *inverse_parser: callable | None = None*, *model_family_keywords: list[str] | None = None*)[[source]](../../_modules/torchrl/data/llm/history.html#add_chat_template)

Add a custom chat template to the global template dictionary.

This function allows you to add custom chat templates for new model families
that support assistant token masking via the {% generation %} keyword.

Parameters:

- **template_name** (*str*) - The name of the template (e.g., "llama", "mistral").
This name will be used in the chat_template_name parameter of
History.apply_chat_template() and History.from_text().
- **template** (*str*) - The Jinja2 template string. Must include {% generation %}
blocks around assistant message content to enable token masking.
- **inverse_parser** (*callable**,**optional*) - A function that parses formatted text back
into a History object. Should have signature (text: str) -> History.
If None, a basic parser will be used.
- **model_family_keywords** (*list**[**str**]**,**optional*) - Keywords to detect this model family
in the auto-detection logic. For example, ["llama", "meta-llama"] for Llama models.
If provided, the template will be automatically selected for models containing
these keywords in their name.

Example

```
>>> from torchrl.data.llm.chat import add_chat_template, History
>>> from transformers import AutoTokenizer
>>>
>>> # Add a custom template for Llama models
>>> llama_template = '''
... {% for message in messages %}
... {%- if message['role'] == 'user' %}
... {{ '<s>[INST] ' + message['content'] + ' [/INST]' }}
... {%- elif message['role'] == 'assistant' %}
... {% generation %}{{ message['content'] + '</s>' }}{% endgeneration %}
... {%- endif %}
... {% endfor %}
... {%- if add_generation_prompt %}
... {% generation %}{{ ' ' }}{% endgeneration %}
... {%- endif %}
... '''
>>>
>>> def parse_llama_text(text: str) -> History:
... # Custom parser for Llama format
... import re
... pattern = r'<s>\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)</s>'
... matches = re.findall(pattern, text, re.DOTALL)
... messages = []
... for user_content, assistant_content in matches:
... messages.append(History(role="user", content=user_content.strip()))
... messages.append(History(role="assistant", content=assistant_content.strip()))
... return lazy_stack(messages)
>>>
>>> # Add the template with auto-detection
>>> add_chat_template(
... template_name="llama",
... template=llama_template,
... inverse_parser=parse_llama_text,
... model_family_keywords=["llama", "meta-llama"]
... )
>>>
>>> # Now you can use it with auto-detection
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> history = History.from_chats([[
... {"role": "user", "content": "Hello"},
... {"role": "assistant", "content": "Hi there!"}
... ]])
>>>
>>> # Auto-detection will use the llama template
>>> result = history.apply_chat_template(
... tokenizer=tokenizer,
... add_generation_prompt=False,
... return_dict=True,
... return_assistant_tokens_mask=True,
... )
>>>
>>> # Or use it explicitly
>>> result = history.apply_chat_template(
... tokenizer=tokenizer,
... chat_template_name="llama",
... add_generation_prompt=False,
... return_dict=True,
... return_assistant_tokens_mask=True,
... )
```