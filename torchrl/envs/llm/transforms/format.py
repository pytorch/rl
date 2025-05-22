# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ===========================================================================
#
# Formatting transforms for LLMs.
#

from __future__ import annotations

from tensordict import NonTensorData

from torchrl.envs.transforms.transforms import Transform


class TemplateTransform(Transform):
    """A transform that maps applies a chat template to an input string during the forward pass, and parses the strings to the template during backward."""

    # alternative to DummyFormat, wip
    def __init__(self, tokenizer, chat_template: str | None = None):
        super().__init__(
            in_keys=["message"],
            out_keys=["message"],
            in_keys_inv=["action"],
            out_keys_inv=["action"],
        )
        if chat_template is None:
            chat_template = tokenizer.get_chat_template()
        self.chat_template = chat_template
        self.tokenizer = tokenizer

    def _apply_transform(self, message):
        if not isinstance(message, str):
            return NonTensorData(
                self._apply_transform(message.data),
                batch_size=message.batch_size,
                device=message.device,
            )
        return self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self.chat_template,
        )

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _inv_apply_transform(self, action):
        if not isinstance(action, str):
            return NonTensorData(
                self._inv_apply_transform(action.data),
                batch_size=action.batch_size,
                device=action.device,
            )
        lines = action.split("\n")

        # Find the last assistant block
        last_assistant_block = []
        for line in reversed(lines):
            if "<|start_header_id|>assistant<|end_header_id|>" in line:
                # last_assistant_block.append(line)
                break
            elif "<|eot_id|>" in line:
                continue
            else:
                last_assistant_block.append(line)

        # Reverse the last assistant block to its original order
        last_assistant_block = list(reversed(last_assistant_block))
        return "\n".join(last_assistant_block)
