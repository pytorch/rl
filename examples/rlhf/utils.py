# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from contextlib import nullcontext

import torch
import torch._dynamo
from hydra.utils import to_absolute_path
from transformers import GenerationConfig, GPT2Tokenizer


def resolve_name_or_path(name_or_path):
    """Hydra changes the working directory, so we need to absolutify paths."""
    if name_or_path.startswith("./") or name_or_path.startswith("/"):
        return to_absolute_path(name_or_path)
    return name_or_path


def get_file_logger(name, filename, level=logging.DEBUG):
    """
    Set up logger that will log to the given filename.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setFormatter(
        # logging.Formatter("%(asctime)s, %(name)s %(levelname)s %(message)s")
        logging.Formatter("%(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup(device, dtype):
    """
    Set manual seed, configure backend and autocasting.
    """
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 256

    if "cuda" not in device:
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))


def flatten_td(td):
    # our tensordict has shape [B, T] where B = batch_size and T = trajectory length
    # some trajectories may have stopped (reached EOS) before generating T tokens
    # this function truncates and concatenates the trajectories, resulting in a
    # tensordict that has shape [N] where N <= B * T.
    done = td["next", "done"]
    mask = torch.zeros_like(done)
    mask[..., 1:, :] = done[..., :-1, :]  # shift by one
    mask = ~mask.cumsum(-2).bool().squeeze()
    return td[mask]


class TestPromptLogger:
    def __init__(self, batch, reward_model, logger, episode_length):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        test_rindex = batch.prompt_rindex[0]
        test_prompt_ids = batch.input_ids[:1, :test_rindex]
        test_label_ids = batch.input_ids[:1, test_rindex:]
        test_prompt = tokenizer.decode(test_prompt_ids[0, :test_rindex].tolist())
        test_label = tokenizer.decode(
            test_label_ids[0, test_label_ids[0] != tokenizer.pad_token_id].tolist()
        )
        _, test_label_reward = reward_model(
            input_ids=batch.input_ids[:1], attention_mask=batch.attention_mask[:1]
        )
        self.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id, max_new_tokens=episode_length
        )
        self.test_prompt_ids = test_prompt_ids
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.test_label_reward = test_label_reward
        self.test_rindex = test_rindex
        self.test_prompt = test_prompt
        self.test_label = test_label
        self.logger = logger

    def log(self, model):
        response_ids = model.generate(
            input_ids=self.test_prompt_ids, generation_config=self.generation_config
        )
        _, response_reward = self.reward_model(
            input_ids=response_ids,
            attention_mask=(response_ids != self.tokenizer.pad_token_id).to(
                torch.int64
            ),
        )
        reward = (response_reward - self.test_label_reward).item()
        response_ids = response_ids[0, self.test_rindex :]
        response = self.tokenizer.decode(
            response_ids[response_ids != self.tokenizer.eos_token_id].tolist()
        )
        string_to_write = (
            f"Query:\n{self.test_prompt}\n"
            f"Response:\n{response}\n"
            f"Actual response:\n{self.test_label}\n"
            f"{reward=:4.4f}\n"
            f"====================================================\n"
        )
        self.logger.info(string_to_write)
