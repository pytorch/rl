from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.data.replay_buffers import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import trange
from transformers import GPT2Tokenizer, GenerationConfig

from data import get_prompt_dataloader
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from utils import get_file_logger, load_config

EOS_TOKEN_ID = 50256


class VmapCritic(TensorDictModuleBase):
    def __init__(self, critic):
        super().__init__()
        self.in_keys = critic.in_keys
        self.out_keys = critic.out_keys
        self.module = critic

    def forward(self, tensordict):
        ndim = tensordict.ndim
        training = self.module.training
        self.module.eval()
        td = torch.vmap(self.module, (ndim - 1,))(tensordict)
        self.module.train(training)
        # vmap sends this dim to the beginning so we need to send it back where it belongs
        td = td.permute(*range(1, ndim), 0)
        return tensordict.update(td)


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


@torch.no_grad()
def generate(model, batch, ref_model, max_new_tokens=50):
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
def create_rollout_td(
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
        torch.tensor(max_new_tokens),
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
    # TODO: add KL penalty in reward
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


def create_loss_estimator(
    config, reward_model, batch, tokenizer, logger=None, ref_model=None
):
    test_rindex = batch.prompt_rindex[0]
    test_prompt_ids = batch.input_ids[:, :test_rindex]
    test_label_ids = batch.input_ids[:, test_rindex:]
    generation_config = GenerationConfig(
        pad_token_id=EOS_TOKEN_ID, max_new_tokens=config["episode_length"]
    )
    test_prompt = tokenizer.decode(test_prompt_ids[0, :test_rindex].tolist())
    test_label = tokenizer.decode(
        test_label_ids[0, test_label_ids[0] != EOS_TOKEN_ID].tolist()
    )
    _, test_label_reward = reward_model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask
    )

    @torch.no_grad()
    def estimate_loss(model, dataloader):
        rewards = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            generated, log_probs, log_ratio = generate(
                model, batch, ref_model=ref_model
            )
            td = create_rollout_td(batch, generated, reward_model, log_probs, log_ratio)
            rewards[k] = td.get(("next", "reward")).sum(dim=1).mean().item()
        test_reward = rewards.mean()

        if logger:
            response_ids = model.generate(
                input_ids=test_prompt_ids, generation_config=generation_config
            )
            _, response_reward = reward_model(
                input_ids=response_ids,
                attention_mask=(response_ids != EOS_TOKEN_ID).to(torch.int64),
            )
            reward = (response_reward - test_label_reward).item()
            response_ids = response_ids[0, test_rindex:]
            response = tokenizer.decode(
                response_ids[response_ids != tokenizer.eos_token_id].tolist()
            )
            string_to_write = (
                f"Query:\n{test_prompt}\n"
                f"Response:\n{response}\n"
                f"Actual response:\n{test_label}\n"
                f"{reward=:4.4f}, "
                f"{test_reward=:4.4f}\n"
                f"====================================================\n"
            )
            logger.debug(string_to_write)

        return test_reward

    return estimate_loss


def main():
    config = load_config("config/train_rlhf.yaml")

    query_logger = get_file_logger("query_logger", "query_logger.log")
    test_reward_logger = get_file_logger("test_reward_logger", "test_rewards.log")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    reward_model = init_reward_model(config)
    actor, critic, critic_head, model = init_actor_critic(config)
    critic.eval()

    ref_model = deepcopy(model).to("cuda:1")
    ref_model.eval()
    ref_model.requires_grad_(False)

    reward_model.eval()
    reward_model.requires_grad_(False)

    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    adv_fn = GAE(
        value_network=VmapCritic(critic), gamma=0.99, lmbda=0.95, average_gae=True
    )

    loss_fn = ClipPPOLoss(actor, critic_head)
    tdl = get_prompt_dataloader(config, split="train")
    vdl = get_prompt_dataloader(config, split="valid")

    test_prompt = next(vdl)[:1]
    estimate_loss = create_loss_estimator(
        config,
        reward_model,
        test_prompt,
        tokenizer,
        logger=query_logger,
        ref_model=ref_model,
    )

    lr = config["learning_rate"]
    wd = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    grad_clip = config["grad_clip"]

    optimizer = torch.optim.AdamW(
        loss_fn.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2)
    )

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(config["episode_length"] * config["num_rollouts"]),
        batch_size=config["ppo_batch_size"],
        sampler=SamplerWithoutReplacement(),
    )
    losses = []
    rewards = []
    test_rewards = []

    for i in trange(config["max_iters"]):
        rb.empty()
        rollout_rewards = []
        for _ in range(0, config["num_rollouts"], config["batch_size"]):
            batch = next(tdl)
            generated, log_probs, log_ratio = generate(model, batch, ref_model=ref_model)
            # generate the tensordict structure expected from a rollout using the generated
            # tokens from the huggingface model
            td = create_rollout_td(batch, generated, reward_model, log_probs, log_ratio)
            with torch.no_grad():
                adv_fn(td)
            # it's possible we didn't fill the replay buffer in the last iteration if
            # generation stopped early, so we empty first before repopulating
            rb.extend(flatten_td(td))
            rollout_rewards.append(td.get(("next", "reward")).mean().cpu().item())
        rewards.append(torch.tensor(rollout_rewards).mean().cpu().item())

        if i % config["eval_interval"] == 0:
            test_rewards.append(estimate_loss(model, vdl))
            test_reward_logger.debug(test_rewards[-1])

        epoch_losses = []
        for epoch in range(config["num_epochs"]):
            for minibatch in rb:
                optimizer.zero_grad()
                loss_vals = loss_fn(minibatch.to(config["device"]))
                loss_val = sum(
                    value for key, value in loss_vals.items() if key.startswith("loss")
                )
                loss_val.backward()
                epoch_losses.append(loss_val.detach().cpu())
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_clip)
                optimizer.step()
        losses.append(torch.tensor(epoch_losses).mean().item())

    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rewards, label="reward")
    ax.plot(losses, label="loss")
    ax.plot(
        torch.arange(0, config["max_iters"], config["eval_interval"]).numpy(),
        test_rewards,
        label="test reward",
    )
    ax.legend()

    f.savefig("figures/rlhf-training-curves.png", dpi=150)

    model.save_pretrained(config["out_dir_rlhf"])


if __name__ == "__main__":
    main()
