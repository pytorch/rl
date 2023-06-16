import torch
import torch.nn.functional as F
import transformers

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
def generate(model: transformers.PreTrainedModel, batch, ref_model: transformers.PreTrainedModel, max_new_tokens=50):
    """Generates a sequence of tokens from a batch of data sampled from the data collector.

    Args:
        model (transformers.Transformer): the model to be used. Should have a
            :meth:`generate` method.
        batch (PromptData): the data to be used.
        ref_model (transformers.Transformer): a frozen version of ``model``
            where params are in their initial configuration.
        max_new_tokens (int, optional): the maximum length of the sequence.
            Defaults to 50.

    Returns:
        generated (torch.Tensor): a [B x (Ti +To)] sequence of integers (tokens),
            where Ti is the length of the input sequence and To is the length
            of the generated sequence.
        log_probs_gen: the log-probabilities of the token generated.
        log_ratio: the log ratio between probabilities under the generative
            model and the frozen version.

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
    attention_mask = (input_ids != EOS_TOKEN_ID).bool()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
    samples = outputs.sequences
    # we'll insert generated tokens into a tensor prepopulated with padding tokens,
    # thereby moving back to right padding for reward model
    mask = samples != EOS_TOKEN_ID
    splits = samples[mask].split(mask.sum(-1).tolist(), 0)
    generated = torch.nested.as_nested_tensor(list(splits))
    generated = torch.nested.to_padded_tensor(generated, EOS_TOKEN_ID)

    # get the scores and normalise for log probabilities
    scores = torch.stack(outputs.scores, 1)
    log_probs_gen = F.pad(
        F.log_softmax(scores, dim=-1).max(dim=-1).values,
        (0, max_new_tokens - scores.shape[1]),
        value=0,
    )

    # get the scores and normalise for log probabilities
    attention_mask = (generated != EOS_TOKEN_ID).bool()
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
    log_ratio = (logprobs - ref_logprobs)
    log_ratio = log_ratio.masked_fill(~attention_mask[:, :-1], 0)
    log_ratio = torch.stack(
        [
            row[rindex - 1 : rindex + max_new_tokens - 1]
            for row, rindex in zip(log_ratio, batch.prompt_rindex)
        ],
        dim=0,
    )
    return generated, log_probs_gen, log_ratio


@torch.no_grad()
def create_rollout_td(
    batch, generated, reward_model, log_probs, log_ratio, max_new_tokens=50, kl_coef=0.1
):
    """A TensorDict wrapper for generated data.

    This function takes a batch plus the generated tokens and replicates the tensordict
    structure that would have been obtained from a rollout with a TorchRL env that
    sampled one token each timestep.

    Args:
        batch:
    """
    rollout_generated = []
    for rindex, row in zip(batch.prompt_rindex, generated):
        arange = torch.arange(row.shape[0], device=generated.device)
        tokens = []
        for i in range(max_new_tokens + 1):
            tokens.append(
                torch.where(
                    arange < rindex + i,
                    row,
                    EOS_TOKEN_ID,
                )
            )
        rollout_generated.append(torch.stack(tokens))
    rollout_generated = torch.stack(rollout_generated)
    rollout_attention_mask = (rollout_generated != EOS_TOKEN_ID).bool()

    # done is True when we either first sample an EOS token or reach the maximum number
    # of generated tokens
    done_idx = torch.minimum(
        (generated != EOS_TOKEN_ID).sum(dim=-1) - batch.prompt_rindex,
        torch.tensor(max_new_tokens),
    )
    done = torch.zeros(done_idx.numel(), max_new_tokens, dtype=torch.bool)
    done = done.scatter(-1, done_idx.unsqueeze(-1), 1)

    # the sequence of actions for each trajectory is just the generated token ids
    action_idx = torch.arange(max_new_tokens, device=generated.device)
    action_idx = action_idx + batch.prompt_rindex.unsqueeze(-1)
    action = generated.gather(-1, action_idx)

    # calculate the reward for the finished sequence
    _, end_scores = reward_model(
        input_ids=rollout_generated[:, -1], attention_mask=rollout_attention_mask[:, -1]
    )
    _, end_scores_labels = reward_model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
    )
    # the reward is zero except for the timestep where we reached a stopping condition
    reward = end_scores - end_scores_labels
    reward = reward.unsqueeze(-1).unsqueeze(-1)
    reward = reward.masked_scatter(~done, 0)
    reward = reward - kl_coef * log_ratio.unsqueeze(-1)
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
    generated, log_probs, log_ratio = generate(model, batch, ref_model=ref_model)
    return create_rollout_td(
        batch, generated, reward_model, log_probs, log_ratio, max_new_tokens, kl_coef
    )
