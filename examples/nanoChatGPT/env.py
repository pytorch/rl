from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from shared import create_infinite_dataloader
from tensordict.prototype import tensorclass
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset

from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
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


def _step(self, tensordict):
    generated = tensordict["generated"]

    # perform the action
    action = tensordict["action"].squeeze(-1)

    # compute the reward
    if generated.shape[-1] >= self.config["episode_length"]:
        reward = self.reward_model(generated).unsqueeze(-1)
        done = torch.ones_like(reward, dtype=torch.bool)
    else:
        reward = torch.zeros((*tensordict.batch_size, 1))
        done = torch.zeros_like(reward, dtype=torch.bool)

    # The output must be written in a ``"next"`` entry
    next_gen = torch.hstack((generated, action[..., None]))
    out = TensorDict(
        {"next": {"generated": next_gen, "reward": reward, "done": done}},
        tensordict.shape,
    )
    return out


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no tensordict is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input tensordict contains all the relevant
        # parameters to get started.
        tensordict = TensorDict({}, batch_size=self.config["batch_size"])

    batch = next(self.dataloader)

    out = TensorDict(
        {
            "generated": batch.prompt[:, -self.config["block_size"] :],
            "done": torch.zeros((*batch.prompt.shape[:-1], 1, 1), dtype=torch.bool),
            "reward": torch.zeros(
                (*batch.prompt.shape[:-1], 1, 1), dtype=torch.float32
            ),
        },
        tensordict.shape,
    )
    return out


def _make_spec(self):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        prompt=UnboundedDiscreteTensorSpec(
            shape=(self.config["batch_size"],),
            dtype=torch.int64,
        ),
        generated=UnboundedDiscreteTensorSpec(
            shape=(self.config["batch_size"],),
            dtype=torch.int64,
        ),
        shape=(self.config["batch_size"],),
    )
    # since the environment is stateless, we expect the previous output as input
    self.input_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec, but the convenient
    # self.action_spec = spec is supported
    self.action_spec = UnboundedDiscreteTensorSpec(
        shape=(self.config["batch_size"], 1),
        dtype=torch.int64,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(
        shape=(self.config["batch_size"], 1, 1)
    )
    self.done_spec = self.reward_spec.clone()


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class RLHFEnv(EnvBase):
    batch_locked = False

    def __init__(self, reward_model=None, config=None, dataloader=None, seed=None):
        # if td_params is None:
        #     td_params = self.gen_params()
        super().__init__(device=config["device"], batch_size=[config["batch_size"]])

        self.reward_model = reward_model
        self.config = config
        self.dataloader = dataloader
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        # self.set_seed(seed)

    # Helpers: _make_step and gen_params
    # gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed


def main():
    config = load_and_update_config("config/train_rl.yaml")
    train_loader, _ = get_dataloaders(config)
    env = RLHFEnv(dataloader=train_loader, config=config)

    td = env.reset()

    def get_action(td):
        print("AAA", td)
        td["action"] = torch.randint(1, 1000, td.shape)
        return td

    # env.rollout(3, get_action, return_contiguous=False)
    print(td.shape)
    print(td)
    print(env.batch_size)
    for i in range(3):
        # td = get_action(td)
        td["action"] = torch.randint(1, 1000, td.shape)
        td = env.step(td)
        print("random step tensordict", i, td["action"], td["next"]["generated"])
        td = step_mdp(td)
    print(list(td.keys(True)))


if __name__ == "__main__":
    main()
