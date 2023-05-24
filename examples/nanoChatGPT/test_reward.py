"""
Train the transformer model. Configurable via config/train.yaml, but any argument can
also be overridden at the command line.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""

import tqdm 
from pathlib import Path
from data import get_reward_dataloaders
from models.reward import init_reward_model
from utils import load_and_update_config

HERE = Path(__file__).parent



def main():
    
    config = load_and_update_config("config/train_reward.yaml")
    config["block_size"] = 512
    # ######## INIT MODELS ########
    model = init_reward_model(config)
    train_loader, val_loader = get_reward_dataloaders(config)

    accuracy = 0
    for i, batch in tqdm.tqdm(enumerate(train_loader)):
        reward_chosen = model(batch.chosen)
        reward_rejected = model(batch.rejected)

        accuracy += (reward_chosen >= reward_rejected).float().mean()

        if i >= 100:
            accuracy /= i
            break

    print(accuracy)


if __name__ == "__main__":
    main()
