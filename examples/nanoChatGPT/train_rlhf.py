import time
from pathlib import Path

import torch

from data.shakespeare import get_dataloaders
from env import RLHFEnv
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from shared import setup
from tensordict.nn import set_skip_existing, TensorDictModuleBase
from torch import vmap

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from utils import load_and_update_config

HERE = Path(__file__).parent


def main():
    config = load_and_update_config("config/train_rlhf.yaml")
    setup(config)

    # ######## INIT MODELS ########
    actor, critic, critic_head = init_actor_critic(config)

    reward_model, _ = init_reward_model(config)

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
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-3)

    # DataLoader
    train_loader, _ = get_dataloaders(config)

    # Environment
    env = RLHFEnv(reward_model=reward_model, config=config, dataloader=train_loader)

    # ######## TRAINING LOOP ########

    ep_length = config["episode_length"]
    max_iters = config["max_iters"]
    num_epochs = config["num_epochs"]
    device = config['device']
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(ep_length * config["batch_size"]),
        batch_size=config["ppo_batch_size"],
        sampler=SamplerWithoutReplacement(),
    )
    rewards = []
    losses = []


    for i in range(max_iters):
        start_time = time.time()
        with torch.no_grad():
            td = env.rollout(ep_length, policy=actor, return_contiguous=True).cpu()
        for epoch in range(num_epochs):
            tdd = adv_fn(td.to(device)).exclude("x")
            # print(tdd)
            rb.extend(tdd.reshape(-1))
            del tdd
            for batch in rb:
                # with set_skip_existing(True):
                loss_vals = loss_fn(batch.to(device))

                loss_val = sum(
                    value for key, value in loss_vals.items() if key.startswith("loss")
                )
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()

        rewards.append(td.get(("next", "reward")).mean().detach().cpu())
        losses.append(loss_val.detach().cpu())
        print(
            f"Iteration {i}: loss={losses[-1]}, "
            f"reward={rewards[-1]: 4.4f}, "
            f"time elapsed={time.time() - start_time:.2f}"
        )

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
