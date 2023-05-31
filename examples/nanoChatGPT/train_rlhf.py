from copy import deepcopy
from pathlib import Path

import tiktoken

import torch
import tqdm

from data.openai_summarize_tldr import get_prompt_dataloaders
from env import RLHFEnv
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from tensordict.nn import TensorDictModuleBase
from torch import vmap
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from utils import get_file_logger, load_and_update_config, setup

HERE = Path(__file__).parent


def evaluate_agent(actor, env, episode_length=50, logger=None):
    enc = tiktoken.get_encoding("gpt2")
    training = actor.training
    actor.eval()
    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        td = env.rollout(episode_length, actor)
    actor.train(training)
    reward = td.get(("next", "reward"))[0, -1].item()
    if logger:
        input_ids = td.get(('next', 'input_ids'))
        mask = td.get(('next', 'attention_mask'))[0, -1] == 1
        prompt = enc.decode(input_ids[0, -1, mask][:-episode_length].tolist())
        response = enc.decode(input_ids[0, -1, mask][-episode_length:].tolist())
        string_to_write = (
            f"Query: \n{prompt},\nResponse: \n{response},\nreward={reward: 4.4f}\n"
            f"====================================================\n"
        )
        logger.debug(string_to_write)
    return reward


def main():
    reward_logger = get_file_logger("reward_logger", "rewards.log")
    test_reward_logger = get_file_logger("test_reward_logger", "test_rewards.log")
    query_logger = get_file_logger("query_logger", "query_logger.log")

    config = load_and_update_config("config/train_rlhf.yaml")
    setup(config)

    # ######## INIT MODELS ########
    actor, critic, critic_head = init_actor_critic(config)
    actor.eval()  # deactivate dropout on all modules
    critic.eval()
    actor2 = deepcopy(actor)
    actor2.eval()
    actor2.requires_grad_(False)

    reward_model = init_reward_model(config)
    reward_model.requires_grad_(False)
    reward_model.eval()

    # ######## INIT TRAINING FUNCTIONS ########
    # Advantage
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
            td = vmap(self.module, (ndim - 1,))(tensordict)
            self.module.train(training)
            # vmap sends this dim to the beginning so we need to send it back where it belongs
            td = td.permute(*range(1, ndim), 0)
            return tensordict.update(td)

    vmap_critic = VmapCritic(critic)

    adv_fn = GAE(value_network=vmap_critic, gamma=0.99, lmbda=0.95, average_gae=True)

    # FIXME: why not using the scheduler?
    # Loss
    loss_fn = ClipPPOLoss(actor, critic_head)

    # Optimizer
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    optimizer = torch.optim.AdamW(
        loss_fn.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2)
    )

    # DataLoader
    train_loader, _ = get_prompt_dataloaders(config)

    # Environment
    env = RLHFEnv(
        reward_model=reward_model,
        config=config,
        dataloader=train_loader,
        ref_model=actor2,
    )

    # Test Environment
    test_config = deepcopy(config)
    test_config["batch_size"] = 1
    test_config["episode_length"] = 50
    train_loader_test, _ = get_prompt_dataloaders(test_config)
    test_env = RLHFEnv(
        reward_model=reward_model,
        config=test_config,
        dataloader=train_loader_test,
        ref_model=actor2,
        test=True,
    )

    # ######## TRAINING LOOP ########

    ep_length = config["episode_length"]
    max_iters = config["max_iters"]
    num_epochs = config["num_epochs"]
    device = config["device"]
    num_envs = config["batch_size"]
    grad_clip = config["grad_clip"]

    total_frames = max_iters * num_envs * ep_length
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(ep_length * num_envs),
        batch_size=config["ppo_batch_size"],
        sampler=SamplerWithoutReplacement(),
    )
    rewards = []
    test_rewards = []
    losses = []

    collector = SyncDataCollector(
        env,
        actor.eval(),
        frames_per_batch=ep_length * num_envs,
        total_frames=total_frames,
        device=device,
        storing_device="cpu",
    )
    pbar = tqdm.tqdm(total=total_frames)
    for i, td in enumerate(collector):
        rewards.append(td.get(("next", "reward")).mean().cpu().item())
        reward_logger.debug(rewards[-1])

        if i % config["eval_interval"] == 0:
            test_rewards.append(
                evaluate_agent(
                    actor,
                    test_env,
                    episode_length=test_config["episode_length"],
                    logger=query_logger,
                )
            )
            test_reward_logger.debug(test_rewards[-1])

        pbar.update(td.numel())
        loss_fn.train()
        for epoch in range(num_epochs):
            with torch.no_grad():
                tdd = td.update(adv_fn(td.select(*adv_fn.in_keys).to(device)).cpu())
            rb.extend(tdd.reshape(-1))
            if len(rb) != tdd.numel():
                raise ValueError(
                    "The replay buffer size and the td content must match "
                    f"exactly, got {len(rb)} and {tdd.numel()} respectively"
                )
            for j, batch in enumerate(rb):
                loss_vals = loss_fn(batch.to(device))

                loss_val = sum(
                    value for key, value in loss_vals.items() if key.startswith("loss")
                )
                loss_val.backward()
                losses.append(loss_val.detach().cpu())
                gn = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(
                    f"Iteration {i}: loss_val={loss_val: 4.4f}, "
                    f"epoch and sub-steps={(epoch, j+1)}, "
                    f"reward={td.get(('next', 'reward')).mean(): 4.4f} (init: {rewards[0]: 4.4f}), "
                    f"grad norm: {gn: 4.4f}"
                )
        actor.eval()
        collector.update_policy_weights_()

    import matplotlib.pyplot as plt

    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rewards, label="reward")
    ax.plot(losses, label="loss")
    ax.legend()

    f.savefig("figures/curves.png", dpi=150)
    # TODO: save model
    # TODO: generate something?


if __name__ == "__main__":
    main()
