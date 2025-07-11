import torch
from torchrl.data.datasets.minari_data  import MinariExperienceReplay 
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs.transforms import Compose, CatTensors, Transform

data = MinariExperienceReplay(

    dataset_id="cartpole/random-v1",
    split_trajs=False,
    root = "/home/jorge/.cache/torchrl/minari",
    batch_size=128,
    sampler=SamplerWithoutReplacement(drop_last=True),
    prefetch=4,
    load_from_local_minari=True,

)


data.sample()