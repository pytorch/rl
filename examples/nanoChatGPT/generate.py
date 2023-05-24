"""
Train the transformer model. Configurable via config/train.yaml, but any argument can
also be overridden at the command line.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
from pathlib import Path
import tiktoken

import torch

from data import get_prompt_dataloaders
from models.transformer import init_transformer
from utils import create_lr_scheduler, load_and_update_config, setup
from transformers import GenerationConfig, RepetitionPenaltyLogitsProcessor

HERE = Path(__file__).parent

def evaluate_agent(actor, env, episode_length=50, logger=None):
    enc = tiktoken.get_encoding("gpt2")
    training = actor.training
    actor.eval()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        td = env.rollout(episode_length, actor)
    actor.train(training)
    reward = td.get(("next", "reward"))[-1, -1].item()
    if logger:
        string_to_write = (
            "Query: \n"
            f"{enc.decode(td.get(('next', 'prompt'))[-1, -1, :-episode_length].tolist())},\n"
            f"Response: \n"
            f"{enc.decode(td.get(('next', 'prompt'))[-1, -1, -episode_length:].tolist())},\n"
            f"reward={reward: 4.4f}\n"
            f"====================================================\n"
        )
        logger.debug(string_to_write)
    return reward


def main():
    
    config = load_and_update_config("config/train_rlhf.yaml")
    config["compile"] = False
    config["device"] = "cpu"
    config["block_size"] = 512
    # ######## INIT MODELS ########
    model = init_transformer(config)
    train_loader, val_loader = get_prompt_dataloaders(config)

    batch = next(iter(train_loader))
    test_prompt = batch.prompt[-1:, :256]
    generation_config = GenerationConfig(
        max_length=512,
        eos_token_id=50256,
        pad_token_id=50256,
    )

    res = model.generate(
        input_ids=test_prompt.to(config["device"]), 
        generation_config=generation_config,
        logits_processor=[RepetitionPenaltyLogitsProcessor(penalty=1.2)],
    )


    enc = tiktoken.get_encoding("gpt2")
    string_to_write = (
        "Query: \n"
        f"{enc.decode(test_prompt[0].tolist())}\n"
        f"Response: \n"
        f"{enc.decode(res[0, len(test_prompt[0]):].tolist())}\n"
        f"====================================================\n"
    )
    print(string_to_write)

if __name__ == "__main__":
    main()
