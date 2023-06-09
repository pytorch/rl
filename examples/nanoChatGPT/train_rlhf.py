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

from data import get_prompt_dataloaders
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from models.transformer import init_transformer
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


@torch.no_grad()
def generate(model, batch, max_new_tokens=50):
    input_ids = batch.transformer_data.input_ids.clone()
    # mask the portion of input_ids that corresponds to the label
    prompt_rindex = batch.transformer_data.prompt_rindex
    label_idx = (
        torch.arange(input_ids.shape[1], device=prompt_rindex.device)
        >= prompt_rindex[:, None]
    )
    input_ids[label_idx] = EOS_TOKEN_ID

    # move padding tokens to left pad
    # huggingface models expect left padding for generation
    input_ids = torch.stack(
        [torch.roll(row, (row == EOS_TOKEN_ID).sum().item(), 0) for row in input_ids]
    )

    # generate and capture scores
    generation_config = GenerationConfig(
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=EOS_TOKEN_ID,
        max_new_tokens=max_new_tokens,
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
    log_probs = F.pad(
        scores.max(dim=-1).values - torch.logsumexp(scores, dim=-1),
        (0, max_new_tokens - scores.shape[1]),
        value=0,
    )
    return generated, log_probs


@torch.no_grad()
def create_rollout_td(batch, generated, reward_model, log_probs, max_new_tokens=50):
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
            for rindex, row in zip(batch.transformer_data.prompt_rindex, generated)
        ],
    )
    rollout_attention_mask = (rollout_generated != EOS_TOKEN_ID).to(torch.int64)

    # done is True when we either first sample an EOS token or reach the maximum number
    # of generated tokens
    done_idx = torch.minimum(
        (generated != EOS_TOKEN_ID).sum(dim=-1) - batch.transformer_data.prompt_rindex,
        torch.tensor(max_new_tokens),
    )
    done = (
        torch.arange(max_new_tokens, device=generated.device) == done_idx[:, None]
    ).unsqueeze(-1)

    # the sequence of actions for each trajectory is just the generated token ids
    action_idx = torch.stack(
        [
            torch.arange(i, i + max_new_tokens, device=generated.device)
            for i in batch.transformer_data.prompt_rindex
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
        input_ids=batch.transformer_data.input_ids,
        attention_mask=batch.transformer_data.attention_mask,
    )
    # TODO: add KL penalty in reward
    # the reward is zero except for the timestep where we reached a stopping condition
    reward = done * (end_scores - end_scores_labels)[:, None, None]
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


def create_loss_estimator(config, reward_model, batch, tokenizer, logger=None):
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
            generated, log_probs = generate(model, batch)
            td = create_rollout_td(batch, generated, reward_model, log_probs)
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

        return reward

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

    adv_fn = GAE(
        value_network=VmapCritic(critic), gamma=0.99, lmbda=0.95, average_gae=True
    )

    loss_fn = ClipPPOLoss(actor, critic_head)
    tdl, vdl = get_prompt_dataloaders(config)

    test_prompt = next(vdl).transformer_data[:1]
    estimate_loss = create_loss_estimator(
        config, reward_model, test_prompt, tokenizer, logger=query_logger
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
        storage=LazyTensorStorage(config["episode_length"] * config["batch_size"]),
        batch_size=config["ppo_batch_size"],
        sampler=SamplerWithoutReplacement(),
    )
    losses = []
    rewards = []
    test_rewards = []

    for i in trange(config["max_iters"]):
        batch = next(tdl)
        generated, log_probs = generate(model, batch)
        # generate the tensordict structure expected from a rollout using the generated
        # tokens from the huggingface model
        td = create_rollout_td(batch, generated, reward_model, log_probs)
        with torch.no_grad():
            adv_fn(td)
        # it's possible we didn't fill the replay buffer in the last iteration if
        # generation stopped early, so we empty first before repopulating
        rewards.append(td.get(("next", "reward")).mean().cpu().item())
        rb.empty()
        rb.extend(flatten_td(td))

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
