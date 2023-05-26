from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .transformer import GPT2
from .utils import crop_block_size, print_trainable_parameters


class RewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = GPT2.from_pretrained(model_path)

        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.transformer = model.transformer
        # replace last layer with the reward layer
        self.lm_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.PAD_ID = GPT2TokenizerFast.from_pretrained("gpt2").eos_token_id

    def forward(
        self, chosen_ids=None, chosen_mask=None, rejected_ids=None, rejected_mask=None
    ):
        chosen_outputs = self.transformer(
            input_ids=chosen_ids, attention_mask=chosen_mask
        )
        chosen_hidden_states = chosen_outputs[0]
        rejected_outputs = self.transformer(
            input_ids=rejected_ids, attention_mask=rejected_mask
        )
        rejected_hidden_states = rejected_outputs[0]

        chosen_rewards = self.lm_head(chosen_hidden_states).squeeze(-1)
        rejected_rewards = self.lm_head(rejected_hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        bs = chosen_ids.shape[0]
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen_ids[i], rejected_ids[i])).item():
                c_inds = (chosen_ids[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_ids.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_ids.shape[1]
            r_inds = (rejected_ids[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_ids.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_ids[i] != rejected_ids[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return loss, chosen_end_scores, rejected_end_scores

    @classmethod
    def from_pretrained(cls, path):
        filename = Path(path) / "reward_model.pt"
        if filename.exists():
            return torch.load(filename)
        return cls(path)

    def save_pretrained(self, path):
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        torch.save(self, save_dir / "reward_model.pt")


def init_reward_model(config):
    if config["init_reward_from"] == "scratch":
        model = RewardModel(config["out_dir"])
    elif config["init_reward_from"] == "resume":
        model = RewardModel.from_pretrained(config["out_dir_reward"])
    else:
        raise ValueError(f"option {config['init_reward_from']=} not recognised")

    # crop down the model block size if desired, using model surgery
    # if config["block_size"] < model.config.n_positions:
    #     print(
    #         f"cropping model from block_size {model.config.n_positions} to {config['block_size']}"
    #     )
    #     crop_block_size(model, config["block_size"])
    #     print_trainable_parameters(model)

    model.to(config["device"])
    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(
        model,
        in_keys=[
            ("batched", "chosen_ids"),
            ("batched", "chosen_mask"),
            ("batched", "rejected_ids"),
            ("batched", "rejected_mask"),
        ],
        out_keys=[
            "loss",
            ("batched", "chosen_end_scores"),
            ("batched", "rejected_end_scores"),
        ],
    )
    return model


# FIXME: out of date
# if __name__ == "__main__":
#     # FIXME: this relative import breaks when running this file
#     # below code gives an example of usage but is not easily runnable
#     from .utils import load_and_update_config

#     enc = tiktoken.get_encoding("gpt2")

#     HERE = Path(__file__).parent
#     config = load_and_update_config(HERE.parent / "config" / "train_reward.yaml")
#     reward_model = init_reward_model(config)

#     prompt = enc.encode("this is a hard-coded prompt!")
#     # add singleton leading dimension to simulate batch dimension
#     prompt = torch.tensor(prompt)[None, :]

#     reward = reward_model.forward_reward(prompt)
#     print(reward)
