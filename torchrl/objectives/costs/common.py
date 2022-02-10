__all__ = ["_LossModule"]

from typing import Iterator, Tuple

import torch
from torch import nn
from torch.nn import Parameter

from torchrl.data.tensordict.tensordict import _TensorDict


class _LossModule(nn.Module):
    def __call__(self, tensordict: _TensorDict) -> _TensorDict:
        raise NotImplementedError

    def _networks(self) -> Iterator[nn.Module]:
        for item in self.__dir__():
            if isinstance(item, nn.Module):
                yield item

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if not name.startswith("_target"):
                yield name, param

    def reset(self) -> None:
        # mainly used for PPO with KL target
        pass
