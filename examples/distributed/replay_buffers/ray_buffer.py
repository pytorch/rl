# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Demonstrates how to use torchrl's RayReplayBuffer to store and sample data across nodes, specifically in the context of Large Language Models (LLMs).

This script showcases a simple producer-consumer setup where one node generates trajectories
from a dialogue dataset and stores them in a shared replay buffer, while another node samples
data from this buffer.

The `Trajectory` class represents a single trajectory, containing prompt, response, tokens,
and other relevant information. The `producer` function generates these trajectories and
extends the replay buffer, while the `consumer` function samples data from the buffer.

The script handles tensors with ragged dimensions. They are stored in lazy stacks of tensordicts (or more specifically,
tensorclasses). Getting the strings returns a list, whereas getting the tensors will raise an error, unless the
format is specified (see examples).

"""

import time
from functools import partial

import ray
import torch
from tensordict import lazy_stack, TensorClass

from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyStackStorage, RayReplayBuffer


class Trajectory(TensorClass["nocast"]):
    # A string or list of strings with the prompts
    prompt: str
    # A string or list of strings with the responses
    response: str
    # A ragged tensor with tokens
    tokens: torch.Tensor
    # A ragged tensor with tokens (responses)
    tokens_response: torch.Tensor
    # A ragged tensor with log-probs (same size as tokens_responses)
    logits: torch.Tensor | None = None
    # A ragged tensor with per-token reward
    rewards: torch.Tensor | None = None


@ray.remote(num_cpus=1)
def producer(rb):
    from datasets import load_dataset

    # Get some tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    dataset = load_dataset("daily_dialog", trust_remote_code=True)["train"]
    data = []
    for i, dialog in enumerate(dataset):
        # Assuming each dialog is a list of utterances
        for j in range(len(dialog["dialog"]) - 1):
            prompt = dialog["dialog"][j]
            response = dialog["dialog"][j + 1]
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            tokens_response = tokenizer.encode(response, return_tensors="pt")
            logits = torch.randn_like(tokens_response, dtype=torch.float16)
            data.append(
                Trajectory(
                    prompt=prompt,
                    response=response,
                    tokens=tokens.squeeze(),
                    tokens_response=tokens_response.squeeze(),
                    logits=logits,
                )
            )
        if i == 256:
            break
    data = lazy_stack(data)
    rb.extend(data)
    torchrl_logger.info(f"Extended with {data=}")
    torchrl_logger.info(f"State of buffer at exit time: {rb}")


@ray.remote(num_cpus=1)
def consumer(rb):
    while not rb.write_count:
        torchrl_logger.info("Consumer waiting for data...")
        time.sleep(1)
    for _ in range(1):
        samples = rb.sample()
        torchrl_logger.info(f"Sampling data: {samples}")
        time.sleep(1)

    # We can also sample fewer elements by passing the batch-size to the sample method
    samples = rb.sample(4)
    # To get the strings, get can use __getitem__
    prompt = samples.prompt
    assert len(prompt) == 4
    assert isinstance(prompt, list)
    response = samples.response
    assert len(response) == 4
    assert isinstance(response, list)
    # For tokens / tokens_response / logits / rewards, we can chose between nested tensors, lists or padded tensors
    tokens_padded = samples.get(
        "tokens", as_padded_tensor=True, padding_value=0, padding_side="right"
    )
    tokens_nested = samples.get("tokens", as_nested_tensor=True, layout=torch.jagged)
    tokens_list = samples.get("tokens", as_list=True)
    torchrl_logger.info(f"{tokens_padded=}")
    torchrl_logger.info(f"{tokens_nested=}")
    torchrl_logger.info(f"{tokens_list=}")
    time.sleep(1)


if __name__ == "__main__":
    # The RB is its own ray worker
    rb = RayReplayBuffer(storage=partial(LazyStackStorage, 1_000_000), batch_size=128)
    # Pass handler to producer
    producer_handler = producer.remote(rb)
    # Pass handler to consumer
    consumer_handler = consumer.remote(rb)
    ray.get([producer_handler, consumer_handler])  # Wait for both tasks to complete
    ray.shutdown()
