from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from transformers import GPT2PreTrainedModel

from .transformer import forward_wrap, init_transformer
from .utils import crop_block_size, print_trainable_parameters


class RewardModel(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.transformer = deepcopy(model.transformer)
        self.block_size = self.transformer.config.n_positions
        # replace last layer with the reward layer
        self.lm_head = nn.Linear(model.lm_head.in_features, 1, bias=False)
        self.config = self.transformer.config

    def forward(self, input_ids):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=None)
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)
        # extract logit of last token in sequence
        return logits[:, -1, :]
    
    @classmethod
    def from_pretrained(cls, path):
        save_dir = Path(path)
        return torch.load(save_dir / "model.pt")


    def save_pretrained(self, path):
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        torch.save(self, save_dir / "model.pt")


def init_reward_model(config):
    model_kwargs = {
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
    }
    if config["init_reward_from"] == "scratch":
        model = init_transformer(
            config, as_tensordictmodule=False, skip_compilation=True
        )
        model = RewardModel(model)
    elif config["init_reward_from"] == "resume":
        model = RewardModel.from_pretrained(config["out_dir_reward"])
    else:
        raise ValueError(f"option {config['init_reward_from']=} not recognised")

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.n_positions:
        print(
            f"cropping model from block_size {model.config.n_positions} to {config['block_size']}"
        )
        crop_block_size(model, config["block_size"])
        print_trainable_parameters(model)

    model.to(config["device"])
    # compile the model
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(model, in_keys=["input"], out_keys=["reward"])
    return model


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
