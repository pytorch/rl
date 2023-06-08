import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, set_skip_existing
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import trange
from transformers import GPT2Tokenizer, GenerationConfig

from data import get_prompt_dataloaders
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from models.transformer import init_transformer
from utils import load_config

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
    label_idx = torch.arange(input_ids.shape[1], device=prompt_rindex.device) >= prompt_rindex[:, None]
    input_ids[label_idx] = EOS_TOKEN_ID

    # move padding tokens to left pad
    # huggingface models expect left padding for generation
    input_ids = torch.stack([torch.roll(row, (row == EOS_TOKEN_ID).sum().item(), 0) for row in input_ids])

    # generate and capture scores
    generation_config = GenerationConfig(
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=EOS_TOKEN_ID,
        max_new_tokens=max_new_tokens,
    )
    outputs = model.generate(
        input_ids=input_ids, attention_mask=(input_ids != EOS_TOKEN_ID).to(torch.int64), generation_config=generation_config
    )
    samples = outputs.sequences
    # we'll insert generated tokens into a tensor prepopulated with padding tokens,
    # thereby moving back to right padding for reward model
    generated = torch.ones_like(input_ids) * EOS_TOKEN_ID
    for i, sample in enumerate(samples):
        mask = sample != EOS_TOKEN_ID
        generated[i, :mask.sum()] = sample[mask]

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
                    torch.where(torch.arange(row.shape[0], device=generated.device) < rindex + i, row, EOS_TOKEN_ID)
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
        (generated != EOS_TOKEN_ID).sum(dim=-1) - batch.transformer_data.prompt_rindex, torch.tensor(max_new_tokens)
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
    print(generated.shape)
    print(f"{action_idx=}")
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
        }
    }
    return TensorDict(td, batch_size=done.shape[:2], device=generated.device)


def flatten_td(td):
    # our tensordict has shape [B, T] where B = batch_size and T = trajectory length
    # some trajectories may have stopped (reached EOS) before generating T tokens
    # this function truncates and concatenates the trajectories, resulting in a
    # tensordict that has shape [N] where N <= B * T.
    done = td["next", "done"]
    mask = torch.zeros_like(done)
    mask[..., 1:, :] = done[..., :-1, : ] # shift by one
    mask = ~mask.cumsum(-2).bool().squeeze()
    return td[mask]


def main():
    config = load_config("config/train_rlhf.yaml")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = init_transformer(config, inference=True)
    reward_model = init_reward_model(config)
    actor, critic, critic_head = init_actor_critic(config)
    critic.eval()

    adv_fn = GAE(
        value_network=VmapCritic(critic), gamma=0.99, lmbda=0.95, average_gae=True
    )

    loss_fn = ClipPPOLoss(actor, critic_head)

    # TODO: setup validation loader for evaluation in training loop
    tdl, _ = get_prompt_dataloaders(config)

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

    for _ in trange(config["max_iters"]):
        batch = next(tdl)
        generated, log_probs = generate(model, batch)
        # generate the tensordict structure expected from a rollout using the generated
        # tokens from the huggingface model
        td = create_rollout_td(batch, generated, reward_model, log_probs)
        with torch.no_grad():
            adv_fn(td)
        # it's possible we didn't fill the replay buffer in the last iteration if
        # generation stopped early, so we empty first before repopulating
        rb.empty()
        rb.extend(flatten_td(td))

        for batch in rb:
            optimizer.zero_grad()
            loss_vals = loss_fn(batch.to(config["device"]))

            loss_val = sum(
                value for key, value in loss_vals.items() if key.startswith("loss")
            )
            loss_val.backward()
            losses.append(loss_val.detach().cpu())
            gn = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_clip)
            optimizer.step()
            # TODO: restore logging / evaluation code

    # TODO: make configurable
    model.save_pretrained("out_rlhf")

if __name__ == "__main__":
    main()
