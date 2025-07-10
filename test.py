import torch
from torchrl.data.datasets.minari_data  import MinariExperienceReplay 
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs.transforms import Compose, CatTensors, Transform


from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    set_lazy_legacy,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)

dataset_id = "atari/skiing/expert-v0"


class PermuteCHW(Transform):
    def _call(self, tensordict):
        obs = tensordict.get("observation")
        print(obs.shape)
        # Convert (H, W, C) -> (C, H, W)
        obs = obs.permute(0, 3, 1, 2) if obs.ndim == 4 else obs.permute(2, 0, 1)
        tensordict.set("observation", obs)
        return tensordict
    
class CNNTransform(Transform):
    """Transform decorator which applies a CNN to the observation."""
    def __init__(self, out_key="observation"):
        
        super().__init__()
        self.out_key = out_key

        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),  
            torch.nn.Flatten(),                  
            torch.nn.Linear(64, 512),
            torch.nn.ReLU(),
        )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        observation = tensordict.get("observation")

        emb_obs = self._cnn(observation)

        tensordict.set("observation", emb_obs)
        return tensordict



if __name__ == "__main__":

    data = MinariExperienceReplay(
        dataset_id=dataset_id,
        split_trajs=False,
        batch_size=128,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
        download=True,

        transform=Compose(
            PermuteCHW(),
            CNNTransform(),
        )
    )