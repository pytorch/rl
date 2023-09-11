# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from transformers import GenerationConfig

EOS_TOKEN_ID = 50256


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


@torch.no_grad()
def _generate(model, batch, ref_model, max_new_tokens=50):
    """Generates responses given a batch of prompts, and computes log probabilities of
    the result in terms of both the model and a reference model.
    """
    input_ids = batch.mask_label().input_ids

    # move padding tokens to left pad
    # huggingface models expect left padding for generation
    input_ids = torch.stack(
        [torch.roll(row, (row == EOS_TOKEN_ID).sum().item(), 0) for row in input_ids]
    )

    # generate and capture scores
    generation_config = GenerationConfig(
        pad_token_id=EOS_TOKEN_ID,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=(input_ids != EOS_TOKEN_ID).to(torch.int64),
        generation_config=generation_config,
    )
    samples = outputs.sequences
    # we'll insert generated tokens into a tensor prepopulated with padding tokens,
    # thereby moving back to right padding for reward model
    generated_shape = torch.Size(
        [input_ids.shape[0], input_ids.shape[1] + max_new_tokens]
    )
    generated = (
        torch.ones(generated_shape, dtype=input_ids.dtype, device=input_ids.device)
        * EOS_TOKEN_ID
    )
    for i, sample in enumerate(samples):
        mask = sample != EOS_TOKEN_ID
        generated[i, : mask.sum()] = sample[mask]

    # get the scores and normalise for log probabilities
    scores = torch.stack(outputs.scores, 1)
    log_probs_gen = F.pad(
        F.log_softmax(scores, dim=-1).max(dim=-1).values,
        (0, max_new_tokens - scores.shape[1]),
        value=0,
    )

    # get the scores and normalise for log probabilities
    attention_mask = (generated != EOS_TOKEN_ID).to(torch.int64)
    logits = model(
        input_ids=generated, attention_mask=attention_mask, return_dict=True
    ).logits
    logprobs = logprobs_of_labels(logits[:, :-1], generated[:, 1:])
    ref_logits = ref_model(
        input_ids=generated.to(ref_model.device),
        attention_mask=attention_mask.to(ref_model.device),
        return_dict=True,
    ).logits.to(logits.device)
    ref_logprobs = logprobs_of_labels(ref_logits[:, :-1], generated[:, 1:])
    log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
    log_ratio = torch.stack(
        [
            row[rindex - 1 : rindex + max_new_tokens - 1]
            for row, rindex in zip(log_ratio, batch.prompt_rindex)
        ],
        dim=0,
    )
    return generated, log_probs_gen, log_ratio


@torch.no_grad()
def _create_rollout_td(
    batch, generated, reward_model, log_probs, log_ratio, max_new_tokens=50, kl_coef=0.1
):
    """
    This function takes a batch plus the generated tokens and replicates the tensordict
    structure that would have been obtained from a rollout with a TorchRL env that
    sampled one token each timestep.
    """
    rollout_generated = torch.stack(
        [
            torch.stack(
                [
                    torch.where(
                        torch.arange(row.shape[0], device=generated.device)
                        < rindex + i,
                        row,
                        EOS_TOKEN_ID,
                    )
                    # + 1 so that we get prompt and full generated sequence as first
                    # and last row respectively
                    for i in range(max_new_tokens + 1)
                ]
            )
            for rindex, row in zip(batch.prompt_rindex, generated)
        ],
    )
    rollout_attention_mask = (rollout_generated != EOS_TOKEN_ID).to(torch.int64)

    # done is True when we either first sample an EOS token or reach the maximum number
    # of generated tokens
    done_idx = torch.minimum(
        (generated != EOS_TOKEN_ID).sum(dim=-1) - batch.prompt_rindex,
        torch.tensor(max_new_tokens) - 1,
    )
    done = (
        torch.arange(max_new_tokens, device=generated.device) == done_idx[:, None]
    ).unsqueeze(-1)

    # the sequence of actions for each trajectory is just the generated token ids
    action_idx = torch.stack(
        [
            torch.arange(i, i + max_new_tokens, device=generated.device)
            for i in batch.prompt_rindex
        ]
    )
    action = generated[
        torch.arange(generated.shape[0], device=generated.device)[:, None],
        action_idx,
    ]

    # calculate the reward for the finished sequence
    _, end_scores = reward_model(
        input_ids=rollout_generated[:, -1], attention_mask=rollout_attention_mask[:, -1]
    )
    _, end_scores_labels = reward_model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
    )
    # the reward is zero except for the timestep where we reached a stopping condition
    reward = (
        done * (end_scores - end_scores_labels)[:, None, None]
        - kl_coef * log_ratio[..., None]
    )
    td = {
        "action": action,
        "input_ids": rollout_generated[:, :-1].clone(),
        "attention_mask": rollout_attention_mask[:, :-1].clone(),
        "sample_log_prob": log_probs,
        "next": {
            "input_ids": rollout_generated[:, 1:].clone(),
            "attention_mask": rollout_attention_mask[:, 1:].clone(),
            "done": done,
            "reward": reward,
        },
    }
    return TensorDict(td, batch_size=done.shape[:2], device=generated.device)


def rollout(batch, model, ref_model, reward_model, max_new_tokens=50, kl_coef=0.1):
    """Perform a rollout.

    This function takes a batch of prompts and performs a rollout:
    - We generate responses from model by feeding it the prompt.
    - We calculate the (normalised) reward of the response from the reward model
    - We calculate the log probabilities of the generated sequence using both model and
      ref_model. The resulting KL divergence is subtracted from the reward

    Args:
        batch: A batch of prompts to use as the basis for generating responses.
        model: A HuggingFace style model that has a `.generate` method.
        ref_model: A reference model to compare against in the KL term.
        reward_model: Model that will compute the reward on the generated response.
        max_new_tokens: Upper limit on the number of tokens that can be generated for
            each prompt.
        kl_coef: coefficient with which to multiply the KL term when calculating the
            reward.
    """
    generated, log_probs, log_ratio = _generate(model, batch, ref_model=ref_model)
    return _create_rollout_td(
        batch, generated, reward_model, log_probs, log_ratio, max_new_tokens, kl_coef
    )
