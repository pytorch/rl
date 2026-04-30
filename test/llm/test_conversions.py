# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from tensordict import lazy_stack, TensorDict
from torchrl.data.llm import History
from torchrl.modules.llm.policies.common import ChatHistory, Text, Tokens

# Test data
SIMPLE_CONVERSATION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

MULTI_TURN_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome!"},
]


@pytest.fixture
def tokenizer():
    """Get a tokenizer for testing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TestChatHistoryConversions:
    """Test conversions from ChatHistory to Text and Tokens."""

    def test_history_to_text_single(self, tokenizer):
        """Test converting a single history to text."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)

        text = chat_history.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.full is not None
        assert isinstance(text.full, list)
        assert len(text.full) == 1
        assert "Hello" in text.full[0]
        assert "Hi there!" in text.full[0]

    def test_history_to_text_batch(self, tokenizer):
        """Test converting a batch of histories to text."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        assert histories.shape[0] == 2
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)

        text = chat_history_batch.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.full is not None
        assert isinstance(text.full, list)
        assert len(text.full) == 2
        assert "Hello" in text.full[0][0]
        assert "helpful assistant" in text.full[1][0]

    def test_history_to_text_prompt_response(self, tokenizer):
        """Test converting history with prompt and response to text."""
        prompt_history = History.from_chats([[SIMPLE_CONVERSATION[0]]])
        full_history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(prompt=prompt_history, full=full_history)

        text = chat_history.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.prompt is not None
        assert text.full is not None
        assert text.response is not None
        assert isinstance(text.prompt, list)
        assert isinstance(text.full, list)
        assert isinstance(text.response, list)
        assert len(text.prompt) == 1
        assert len(text.full) == 1
        assert len(text.response) == 1
        # Response should be the part after prompt
        assert text.full[0].startswith(text.prompt[0])
        assert text.full[0] == text.prompt[0] + text.response[0]

    def test_history_to_tokens_single(self, tokenizer):
        """Test converting a single history to tokens."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)

        tokens = chat_history.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert tokens.full is not None
        # Check if it's a nested tensor by checking if it has the _values attribute
        assert hasattr(tokens.full, "_values") or isinstance(tokens.full, torch.Tensor)
        assert tokens.padded is False

    def test_history_to_tokens_batch(self, tokenizer):
        """Test converting a batch of histories to tokens."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)

        tokens = chat_history_batch.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert (full := tokens.get("full", as_nested_tensor=True)) is not None
        # Check if it's a nested tensor
        assert hasattr(full, "_values") or isinstance(full, torch.Tensor)
        assert not any(tokens.padded)
        # Check batch size
        assert tokens.batch_size[0] == 2

    def test_history_to_tokens_prompt_response(self, tokenizer):
        """Test converting history with prompt and response to tokens."""
        prompt_history = History.from_chats([[SIMPLE_CONVERSATION[0]]])
        full_history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(prompt=prompt_history, full=full_history)

        tokens = chat_history.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert tokens.prompt is not None
        assert tokens.full is not None
        assert tokens.response is not None
        # Check if they're nested tensors
        assert hasattr(tokens.prompt, "_values") or isinstance(
            tokens.prompt, torch.Tensor
        )
        assert hasattr(tokens.full, "_values") or isinstance(tokens.full, torch.Tensor)
        assert hasattr(tokens.response, "_values") or isinstance(
            tokens.response, torch.Tensor
        )
        # Response should be the part after prompt
        prompt_tokens_list = tokens._tensordict.get("prompt", as_list=True)
        full_tokens_list = tokens._tensordict.get("full", as_list=True)
        response_tokens_list = tokens._tensordict.get("response", as_list=True)
        prompt_len = prompt_tokens_list[0].shape[0]
        full_len = full_tokens_list[0].shape[0]
        response_len = response_tokens_list[0].shape[0]
        assert full_len == prompt_len + response_len


class TestTokensConversions:
    """Test conversions from Tokens to Text and ChatHistory."""

    def test_tokens_to_text_single(self, tokenizer):
        """Test converting tokens to text."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)
        tokens = chat_history.to_tokens(tokenizer)

        text = tokens.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.full is not None
        assert isinstance(text.full, list)
        assert len(text.full) == 1

    def test_tokens_to_text_batch(self, tokenizer):
        """Test converting a batch of tokens to text."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)
        tokens = chat_history_batch.to_tokens(tokenizer)

        text = tokens.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.full is not None
        assert isinstance(text.full, list)
        assert len(text.full) == 2

    def test_tokens_to_text_prompt_response(self, tokenizer):
        """Test converting tokens with prompt and response to text."""
        prompt_history = History.from_chats([[SIMPLE_CONVERSATION[0]]])
        full_history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(prompt=prompt_history, full=full_history)
        tokens = chat_history.to_tokens(tokenizer)

        text = tokens.to_text(tokenizer)

        assert isinstance(text, Text)
        assert text.prompt is not None
        assert text.full is not None
        assert text.response is not None

    def test_tokens_to_text_padded_error(self, tokenizer):
        """Test that padded tokens raise an error."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)
        tokens = chat_history.to_tokens(tokenizer)
        tokens.padded = True  # Manually set to padded

        with pytest.raises(ValueError, match="padded tokens"):
            tokens.to_text(tokenizer)

    def test_tokens_to_history_single(self, tokenizer):
        """Test converting tokens to history."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)
        tokens = chat_history.to_tokens(tokenizer)

        reconstructed_history = tokens.to_history(tokenizer)

        assert isinstance(reconstructed_history, ChatHistory)
        assert reconstructed_history.full is not None

    def test_tokens_to_history_batch(self, tokenizer):
        """Test converting a batch of tokens to history."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)
        tokens = chat_history_batch.to_tokens(tokenizer)

        reconstructed_history = tokens.to_history(tokenizer)

        assert isinstance(reconstructed_history, ChatHistory)
        assert reconstructed_history.full is not None
        assert reconstructed_history.batch_size[0] == 2


class TestTextConversions:
    """Test conversions from Text to Tokens and ChatHistory."""

    def test_text_to_tokens_single(self, tokenizer):
        """Test converting text to tokens."""
        text_obj = Text(full=["Hello, how are you?"])

        tokens = text_obj.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert tokens.full is not None
        assert isinstance(tokens.full, torch.Tensor)
        assert tokens.padded is False

    def test_text_to_tokens_batch(self, tokenizer):
        """Test converting a batch of text to tokens."""
        text_obj = Text._from_tensordict(TensorDict(batch_size=(2,)).to_lazystack(0))
        with text_obj.view(-1) as text_flat:
            text_flat.full = ["Hello, how are you?", "I'm doing great!"]

        tokens = text_obj.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert (full := tokens.get("full", as_nested_tensor=True)) is not None
        assert isinstance(full, torch.Tensor)
        assert tokens.batch_size[0] == 2

    def test_text_to_tokens_prompt_response(self, tokenizer):
        """Test converting text with prompt and response to tokens."""
        text_obj = Text(
            prompt=["Hello"],
            response=[", how are you?"],
            full=["Hello, how are you?"],
        )

        tokens = text_obj.to_tokens(tokenizer)

        assert isinstance(tokens, Tokens)
        assert tokens.prompt is not None
        assert tokens.response is not None
        assert tokens.full is not None

    def test_text_to_tokens_padding_error(self, tokenizer):
        """Test that padding raises an error."""
        text_obj = Text(full=["Hello, how are you?"])

        with pytest.raises(ValueError, match="Padding is not yet supported"):
            text_obj.to_tokens(tokenizer, padding=True)

    def test_text_to_history_single(self, tokenizer):
        """Test converting text to history."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)
        text_obj = chat_history.to_text(tokenizer)

        reconstructed_history = text_obj.to_history(tokenizer)

        assert isinstance(reconstructed_history, ChatHistory)
        assert reconstructed_history.full is not None

    def test_text_to_history_batch(self, tokenizer):
        """Test converting a batch of text to history."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)
        text_obj = chat_history_batch.to_text(tokenizer)

        reconstructed_history = text_obj.to_history(tokenizer)

        assert isinstance(reconstructed_history, ChatHistory)
        assert reconstructed_history.full is not None
        assert reconstructed_history.batch_size[0] == 2


class TestBijectivity:
    """Test that conversions are bijective (round-trip conversions)."""

    def test_history_to_text_to_history(self, tokenizer):
        """Test History -> Text -> History round-trip."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)

        # Convert to text and back
        text = chat_history.to_text(tokenizer)
        reconstructed = text.to_history(tokenizer)

        assert isinstance(reconstructed, ChatHistory)
        assert reconstructed.full is not None
        # Check that the content is preserved
        original_content = history.content
        reconstructed_content = reconstructed.full.content
        assert original_content == reconstructed_content

    def test_history_to_tokens_to_text_to_history(self, tokenizer):
        """Test History -> Tokens -> Text -> History round-trip."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)

        # Convert through tokens and text
        tokens = chat_history.to_tokens(tokenizer)
        text = tokens.to_text(tokenizer)
        reconstructed = text.to_history(tokenizer)

        assert isinstance(reconstructed, ChatHistory)
        assert reconstructed.full is not None
        # Check that the content is preserved
        original_content = [msg.content for msg in history.unbind(0)[0].unbind(0)]
        reconstructed_content = [
            msg.content for msg in reconstructed.full.unbind(0)[0].unbind(0)
        ]
        assert original_content == reconstructed_content

    def test_text_to_tokens_to_text(self, tokenizer):
        """Test Text -> Tokens -> Text round-trip."""
        original_text = Text(full=["Hello, how are you?"])

        # Convert to tokens and back
        tokens = original_text.to_tokens(tokenizer)
        reconstructed_text = tokens.to_text(tokenizer)

        assert isinstance(reconstructed_text, Text)
        assert reconstructed_text.full is not None
        # The text should be very similar (may have minor tokenization artifacts)
        reconstructed_full_list = reconstructed_text._tensordict.get(
            "full", as_list=True
        )
        original_full_list = original_text._tensordict.get("full", as_list=True)
        assert len(reconstructed_full_list) == len(original_full_list)

    def test_tokens_to_text_to_tokens_shape_preserved(self, tokenizer):
        """Test that Tokens -> Text -> Tokens preserves token shapes."""
        history = History.from_chats([SIMPLE_CONVERSATION])
        chat_history = ChatHistory(full=history)
        original_tokens = chat_history.to_tokens(tokenizer)

        # Convert to text and back to tokens
        text = original_tokens.to_text(tokenizer)
        reconstructed_tokens = text.to_tokens(tokenizer)

        assert isinstance(reconstructed_tokens, Tokens)
        assert reconstructed_tokens.full is not None
        # Check that shapes are similar (may differ slightly due to tokenization)
        original_full_list = original_tokens._tensordict.get("full", as_list=True)
        reconstructed_full_list = reconstructed_tokens._tensordict.get(
            "full", as_list=True
        )
        original_len = original_full_list[0].shape[0]
        reconstructed_len = reconstructed_full_list[0].shape[0]
        # Allow some tolerance for tokenization differences
        assert abs(original_len - reconstructed_len) <= 2


class TestBatchDimensions:
    """Test that conversions work correctly with different batch dimensions."""

    def test_single_batch_dimension(self, tokenizer):
        """Test conversions with single batch dimension."""
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a batch of ChatHistory objects
        chat_histories = [
            ChatHistory(full=histories[i : i + 1]) for i in range(histories.shape[0])
        ]
        chat_history_batch = lazy_stack(chat_histories)
        assert chat_history_batch.batch_size == torch.Size([2])

        # Test all conversions maintain batch size
        text = chat_history_batch.to_text(tokenizer)
        assert text.batch_size == torch.Size([2])

        tokens = chat_history_batch.to_tokens(tokenizer)
        assert tokens.batch_size == torch.Size([2])

        reconstructed = text.to_history(tokenizer)
        assert reconstructed.batch_size == torch.Size([2])

    def test_nested_batch_dimensions(self, tokenizer):
        """Test conversions with nested batch dimensions."""
        # Create a 2x2 batch
        histories = History.from_chats([SIMPLE_CONVERSATION, MULTI_TURN_CONVERSATION])
        # Create a 2x2 batch of ChatHistory objects
        chat_histories_outer = []
        for _ in range(2):
            chat_histories_inner = [
                ChatHistory(full=histories[i : i + 1])
                for i in range(histories.shape[0])
            ]
            chat_histories_outer.append(lazy_stack(chat_histories_inner))
        chat_history_batch = lazy_stack(chat_histories_outer)
        assert chat_history_batch.batch_size == torch.Size([2, 2])

        # Test conversions maintain batch size
        text = chat_history_batch.to_text(tokenizer)
        assert text.batch_size == torch.Size([2, 2])

        tokens = chat_history_batch.to_tokens(tokenizer)
        assert tokens.batch_size == torch.Size([2, 2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()
    pytest.main([__file__, "-v"] + args)
