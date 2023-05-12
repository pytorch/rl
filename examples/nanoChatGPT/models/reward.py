from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from .transformer import init_transformer
from .utils import _remove_state_dict_prefixes, load_checkpoint


class RewardModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = deepcopy(model)
        self.config = model.config

        self.n_embd = model.lm_head.in_features
        self.block_size = model.config.block_size
        self.reward_head = nn.Linear(self.model.lm_head.in_features, 1, bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.model.transformer.wte(idx)
        # position embeddings of shape (1, t, n_embd)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)

        return self.reward_head(x[:, -1, :])


def init_reward_model(config):
    # skip compilation because we will compile the entire reward model as one
    model, model_kwargs = init_transformer(
        config, as_tensordictmodule=False, skip_compilation=True
    )
    model = RewardModel(model)

    print("Config of model: ", model.config)
    out_dir = Path(config["out_dir_reward"])

    if not out_dir.exists():
        print(f"Create {config['out_dir_reward']}")
        out_dir.mkdir()

    if config["init_reward_from"] == "scratch":
        print("initializing reward from scratch")
    elif config["init_reward_from"] == "resume":
        print(f"Resuming training from {config['out_dir_reward']}")
        checkpoint = load_checkpoint(out_dir, device=config["device"])
        state_dict = checkpoint["model"]
        _remove_state_dict_prefixes(state_dict, unwanted_prefixes=["_orig_mod."])
        model.load_state_dict(state_dict)

    model.to(config["device"])
    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(model, in_keys=["input"], out_keys=["reward"])
    return model, model_kwargs


if __name__ == "__main__":
    import tiktoken

    # FIXME: this relative import breaks when running this file
    # below code gives an example of usage but is not easily runnable
    from .utils import load_and_update_config

    enc = tiktoken.get_encoding("gpt2")

    HERE = Path(__file__).parent
    config = load_and_update_config(HERE.parent / "config" / "train_reward.yaml")
    reward_model = init_reward_model(config)

    prompt = enc.encode("this is a hard-coded prompt!")
    # add singleton leading dimension to simulate batch dimension
    prompt = torch.tensor(prompt)[None, :]

    reward = reward_model.forward_reward(prompt)
    print(reward)
