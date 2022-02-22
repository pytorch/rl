__all__ = ["_LossModule"]

from typing import Iterator, Tuple

import torch
from torch import nn
from torch.nn import Parameter

from torchrl.data.tensordict.tensordict import _TensorDict


class _LossModule(nn.Module):
    """
    A parent class for RL losses.
    _LossModule inherits from nn.Module. It is designed to read an input TensorDict and return another tensordict
    with loss keys named "loss_*".
    Splitting the loss in its component can then be used by the agent to log the various loss values throughout
    training. Other scalars present in the output tensordict will be logged too.
    """
    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        """
        It is designed to read an input TensorDict and return another tensordict
        with loss keys named "loss*".
        Splitting the loss in its component can then be used by the agent to log the various loss values throughout
        training. Other scalars present in the output tensordict will be logged too.

        Args:
            tensordict: an input tensordict with the values required to compute the loss.

        Returns:
            A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
            is essential that the losses are returned with this name as they will be read by the agent before
            backpropagation.
        """
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
