from pathlib import Path

import torch
from env import RLHFEnv
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model
from shared import setup
from tensordict.nn import set_skip_existing
from utils import load_and_update_config

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from data.shakespeare import get_dataloaders

HERE = Path(__file__).parent


def main():
    config = load_and_update_config("config/train_rlhf.yaml")
    setup(config)

    # ######## INIT MODELS ########
    actor, critic = init_actor_critic(config)

    reward_model, _ = init_reward_model(config)

    # ######## INIT TRAINING FUNCTIONS ########
    # Advantage
    adv_fn = GAE(value_network=critic, gamma=0.99, lmbda=0.95, average_gae=True)
    # FIXME: why not using the scheduler?
    # Loss
    loss_fn = ClipPPOLoss(actor, critic, gamma=0.99)

    # Optimizer
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-3)

    # DataLoader
    train_loader, _ = get_dataloaders(config)

    # Environment
    env = RLHFEnv(reward_model=reward_model, config=config, dataloader=train_loader)

    # ######## TRAINING LOOP ########

    def get_action(td):
        critic(td)
        actor(td)
        td["sample_log_prob"] = td["sample_log_prob"].detach()
        return td

    for i in range(config["max_iters"]):
        td = env.rollout(
            config["episode_length"], policy=get_action, return_contiguous=False
        )

        # TODO: add replay buffer?
        with set_skip_existing(True):
            adv_fn(td)
            loss_vals = loss_fn(td)

        loss_val = sum(
            value for key, value in loss_vals.items() if key.startswith("loss")
        )
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        print(f"Iteration {i}: {loss_val=}")

    # TODO: save model
    # TODO: generate something?

if __name__ == "__main__":
    main()
