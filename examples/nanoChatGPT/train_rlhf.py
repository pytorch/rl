import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from env import RLHFEnv
from model import RLHF
from shared import create_infinite_dataloader, init_model, setup
from tensordict.nn import set_skip_existing, TensorDictModule
from tensordict.prototype import tensorclass
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset

from torchrl.modules import (
    ActorValueOperator,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from utils import load_and_update_config

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0).contiguous()
        batch.batch_size = []
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PairedDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitem__(self, idx):
        return Data(
            prompt=torch.from_numpy(
                self._memmap[idx : idx + self.block_size].astype(np.int64)
            ),
            target=torch.from_numpy(
                self._memmap[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            ),
            batch_size=[self.block_size],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size


def create_datasets(config):
    data_dir = HERE / "nanoGPT" / "data" / config["dataset"]
    train_data = PairedDataset(data_dir / "train.bin", block_size=config["block_size"])
    val_data = PairedDataset(data_dir / "val.bin", block_size=config["block_size"])

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader


class ActorCritic(ActorValueOperator):
    def __init__(self, base_model):

        base_model = copy.deepcopy(base_model)
        n_embd = base_model.lm_head.in_features

        # actor network
        # extract last layer to be reused by actor
        actor_head = base_model.lm_head
        base_model.lm_head = nn.Identity()

        # critic network
        value_head = nn.Linear(n_embd, 1, bias=False)

        common = TensorDictModule(base_model, in_keys=["prompt"], out_keys=["x"])

        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["logits"])
        actor_head = SafeProbabilisticTensorDictSequential(
            actor_head,
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=Categorical,
                return_log_prob=True,
            ),
        )
        value_head = TensorDictModule(
            value_head, in_keys=["x"], out_keys=["state_value"]
        )

        super().__init__(common, actor_head, value_head)


def train(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls

    # model init: Actor Critic
    # FIXME: Don't like this. include it into model
    model_base, _ = init_model(config)
    a2c_model = ActorCritic(model_base)
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()

    # model init: Reward
    reward_model = RLHF(model_base, "reward", discrete_reward=config["discrete_reward"])

    # TODO: update reward model weights here?
    ckpt_path = os.path.join(config["out_dir_reward"], "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=config["device"])
    # TODO: put this in a shared function
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefixes = ["_orig_mod."]
    for unwanted_prefix in unwanted_prefixes:
        for k in list(state_dict):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    reward_model.load_state_dict(state_dict)

    reward_model = TensorDictModule(
        reward_model, in_keys=["input"], out_keys=["reward"]
    )

    # MODEL TO DEVICE
    reward_model.to(config["device"])
    a2c_model.to(config["device"])

    adv_fn = GAE(value_network=critic, gamma=0.99, lmbda=0.95, average_gae=True)
    loss_fn = ClipPPOLoss(actor, critic, gamma=0.99)

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-3)

    train_loader, _ = get_dataloaders(config)
    max_iters = 100_000

    env = RLHFEnv(reward_model=reward_model, config=config, dataloader=train_loader)

    def get_action(td):
        prompt = td["generated"]
        td["x"], td["state_value"] = critic(prompt)
        _, _, td["action"], td["sample_log_prob"] = actor(prompt)
        td["sample_log_prob"] = td["sample_log_prob"].detach()
        return td

    for i in range(max_iters):
        td = env.rollout(
            config["episode_length"], policy=get_action, return_contiguous=False
        )

        with set_skip_existing(True):
            adv_fn(td)
            loss_vals = loss_fn(td)

        loss_val = sum(
            value for key, value in loss_vals.items() if key.startswith("loss")
        )
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i}: {loss_val=}")


PPO_CONFIG = {
    # cardinality of the sub-samples gathered from the current data in the inner loop
    "sub_batch_size": 64,
    # optimisation steps per batch of data collected
    "num_epochs": 10,
    # clip value for PPO loss: see the equation in the intro for more context.
    "clip_epsilon": (0.2),
    "gamma": 0.99,
    "lmbda": 0.95,
    "entropy_eps": 1e-4,
}


def main():
    config = load_and_update_config("config/train_rl.yaml")
    config["ppo"] = PPO_CONFIG

    setup(config)
    train(config)


if __name__ == "__main__":
    main()
