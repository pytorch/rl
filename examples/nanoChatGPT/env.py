from pathlib import Path
from typing import Optional

import torch

from data.openai_summarize_tldr import get_prompt_dataloaders
from models.reward import init_reward_model
from models.transformer import DEFAULT_VOCAB_SIZE
from tensordict.tensordict import TensorDict

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
from utils import load_and_update_config

HERE = Path(__file__).parent


@torch.no_grad()
def _step(self, tensordict):
    self.step_num += 1
    input_ids = tensordict.get("input_ids")
    attention_mask = tensordict.get("attention_mask")

    # perform the action
    action = tensordict.get("action")

    # The output must be written in a ``"next"`` entry
    next_input_ids = input_ids.clone()
    idx = torch.arange(input_ids.shape[0], device=input_ids.device)
    pad_index = attention_mask.argmin(dim=1)
    next_input_ids[idx, pad_index] = action
    next_attention_mask = attention_mask.clone()
    next_attention_mask[idx, pad_index] = 1

    # compute the reward
    if self.step_num >= self.episode_length:
        _, reward = self.reward_model(next_input_ids, next_attention_mask)
        reward = reward.unsqueeze(-1)
        done = torch.ones_like(reward, dtype=torch.bool)
    else:
        reward = torch.zeros((*tensordict.batch_size, 1)).to(input_ids.device)
        done = torch.zeros_like(reward, dtype=torch.bool)
    assert self.reward_spec.shape == reward.shape, (
        self.batch_size,
        reward.shape,
        reward.dtype,
    )
    assert self.done_spec.shape == done.shape, (self.batch_size, done.shape, done.dtype)

    # compute KL divergence component to avoid diverging too much from original model
    if self.ref_model:
        logits = tensordict.get("sample_log_prob")
        ref_logits = self.ref_model(input_ids, attention_mask)
        kl_penalty = self.kl_th * (logits - ref_logits).mean(-1).unsqueeze(-1)

    reward = reward - self.normalized_reward - kl_penalty

    out = TensorDict(
        {
            "next": {
                "input_ids": next_input_ids,
                "attention_mask": next_attention_mask,
                "reward": reward,
                "done": done,
            }
        },
        tensordict.shape,
    )
    return out


@torch.no_grad()
def _reset(self, tensordict):
    self.step_num = 0
    batch = next(self.dataloader)

    if self.eval_prompt is None:
        self.eval_prompt = (
            batch.transformer_data.input_ids[0],
            batch.transformer_data.attention_mask[0],
        )
    else:
        (
            batch.transformer_data.input_ids[0],
            batch.transformer_data.attention_mask[0],
        ) = self.eval_prompt

    masked_batch = batch.mask_label()
    _, self.normalized_reward = self.reward_model(
        masked_batch.transformer_data.input_ids,
        masked_batch.transformer_data.attention_mask,
    )
    self.normalized_reward = self.normalized_reward.unsqueeze(-1)

    out = TensorDict(
        {
            "input_ids": masked_batch.transformer_data.input_ids,
            "attention_mask": masked_batch.transformer_data.attention_mask,
            "done": torch.zeros(
                (*masked_batch.transformer_data.input_ids.shape[:-1], 1), dtype=torch.bool
            ),
        },
        self.batch_size,
    )
    return out


def _make_spec(self):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        input_ids=BoundedTensorSpec(
            minimum=0,
            maximum=DEFAULT_VOCAB_SIZE,
            shape=(*self.batch_size, self.config["block_size"]),
            dtype=torch.int64,
        ),
        attention_mask=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=(*self.batch_size, self.config["block_size"]),
            dtype=torch.int64,
        ),
        shape=self.batch_size,
    )
    # since the environment is stateless, we expect the previous output as input
    self.input_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec, but the convenient
    # self.action_spec = spec is supported
    self.action_spec = BoundedTensorSpec(
        minimum=0,
        maximum=DEFAULT_VOCAB_SIZE,
        shape=(*self.batch_size, 1),
        dtype=torch.int64,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1))
    self.done_spec = BoundedTensorSpec(
        minimum=0,
        maximum=1,
        shape=(*self.batch_size, 1),
        dtype=torch.bool,
    )


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class RLHFEnv(EnvBase):
    batch_locked = False

    def __init__(
        self, reward_model=None, config=None, dataloader=None, seed=None, ref_model=None
    ):
        # if td_params is None:
        #     td_params = self.gen_params()
        batch_size = config["batch_size"]
        device = config["device"]
        self.episode_length = config["episode_length"]
        self.block_size = config["block_size"]
        super().__init__(device=device, batch_size=[batch_size])

        self.reward_model = reward_model
        self.config = config
        self.dataloader = dataloader
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.step_num = 0
        self.ref_model = ref_model
        self.ref_model.select_out_keys("sample_log_prob")
        self.kl_th = 0.1
        self.eval_prompt = None
        # self.set_seed(seed)

    # Helpers: _make_step and gen_params
    # gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed


def main():
    from torchrl.envs import check_env_specs

    config = load_and_update_config("config/train_rlhf.yaml")
    reward_model = init_reward_model(config)
    train_loader, _ = get_prompt_dataloaders(config)
    env = RLHFEnv(reward_model=reward_model, dataloader=train_loader, config=config)

    check_env_specs(env)

    td = env.reset()

    def get_action(td):
        td["action"] = torch.randint(1, 1000, td.shape, device=config["device"])
        return td

    # rollout
    env.rollout(3, get_action, return_contiguous=False)

    print(td.shape)
    print(td)
    print(env.batch_size)
    # manual rollout
    for i in range(3):
        # td = get_action(td)
        td["action"] = torch.randint(1, 1000, td.shape)
        td = env.step(td)
        print("random step tensordict", i, td["action"], td["next", "prompt"])
        td = step_mdp(td)
    print(list(td.keys(True)))


if __name__ == "__main__":
    main()
