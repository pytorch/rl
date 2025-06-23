# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase, lazy_stack
from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.envs.transforms import Transform
from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.writers import RoundRobinWriter

from typing import Literal

class AcceptanceRewardSelector(Transform):
    """A replay-buffer transform that marks items as accepted or rejected, based on a reward threshold.

    Args:
        reward_threshold (float | Literal["mean", "median"]): Threshold for the reward to be considered accepted.
            Can be a `float` value or `"mean"` or `"median"`, in which case the acceptance is based on the mean or median of the rewards
            over cumulated batches (`total = total_dialog_turns`).
    
    Keyword Args:
        total_dialog_turns (int): Number of dialog turns to keep in memory for the acceptance selection.
        reward_key (NestedKey): Key to the reward in the tensordict. Defaults to ("next", "reward").
        done_key (NestedKey): Key to the done state in the tensordict. Defaults to ("next", "done").
        accept_key (NestedKey): Key to the accept state in the tensordict. Defaults to ("next", "is_chosen").
        verbose (bool): Whether to print verbose information. Defaults to `False`.

    """

    def __init__(
        self,
        reward_threshold: float | Literal["mean", "median"],
        *,
        total_dialog_turns: int,
        reward_key: NestedKey = ("next", "reward"),
        done_key: NestedKey = ("next", "done"),
        accept_key: NestedKey = ("next", "is_chosen"),
        prompt_key: NestedKey = "text",
        verbose: bool = False,
    ):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.total_dialog_turns = total_dialog_turns
        self.queues = defaultdict(deque)
        self._cumul = isinstance(reward_threshold, str)

        self.reward_key = reward_key
        self.done_key = done_key
        self.accept_key = accept_key
        self.prompt_key = prompt_key
        self.verbose = verbose
 
    def forward(self, tensordict: TensorDictBase) -> Any:
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # flip batch size and accept/reject dim to have a TD of shape [2, batch_size]
        tensordict = tensordict.transpose(1, 0)
        if tensordict.shape[0] != 2:
            raise ValueError(f"Expected a TD of shape [2, batch_size], got {tensordict.shape=}")
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # This transform expects trajectories, either in batches or a single (cat of) trajectories
        if tensordict.ndim == 1:
            # Check how many done states we have
            num_done = tensordict[self.done_key].sum()
            if num_done > 1:
                done_idx = tensordict[self.done_key].nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                tensordicts = tensordict.split(splits)
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts, 0) if tensordicts else None
            # Then we have a single trajectory. Check if it's done
            if not tensordict[-1][self.done_key].all():
                raise RuntimeError("Expected the trajectory to be done.")
            # Now we have a single, done trajectory. Get the prompt, add it to the corresponding queue
            prompt = tensordict[0][self.prompt_key]
            if not isinstance(prompt, str):
                raise TypeError(f"Expected a string as prompt, got {type(prompt)=}")
            self.queues[prompt].append(tensordict)
            # If the queue is full, we can process it and pass it to the buffer
            
            if len(self.queues[prompt]) == self.total_dialog_turns:
                if self.verbose:
                    torchrl_logger.info(f"Getting top-k rewards for {prompt=}")
                # lazy_stack of the trajectories
                tds = lazy_stack(list(self.queues.pop(prompt)), 0)
                # Collect rewards: they will have shape (total_dialog_turns, traj_len, *reward_shape)
                reward = tds.get(self.reward_key, as_nested_tensor=True)
                print(f"{reward=}")
                reward = self._aggregate_rewards(reward)
                # Check if all rewards are equal
                if (reward == reward[0]).all():
                    # If all rewards are equal, we can't select top-k - discard the trajectories
                    if self.verbose:
                        torchrl_logger.warning(
                            f"All rewards are equal ({reward.unique()=})"
                        )
                    return
                # Filter out rewards below median / target value
                if self.reward_threshold == "median":
                    reward_threshold = reward.median(dim=-1, keepdim=True)[0]
                elif self.reward_threshold == "mean":
                    reward_threshold = reward.mean(dim=-1, keepdim=True)[0]
                else:
                    reward_threshold = self.reward_threshold
                mask = reward > reward_threshold
                try:
                    tds.set(self.accept_key, mask.view(tds.shape))
                except Exception as e:
                    raise RuntimeError(f"Failed setting the accept key with shape {mask.shape} for {tds.shape=}. It is expected that the number of elements of the accept key is the same as the number of elements in the tensordict.") from e
                accepted_tds = tds[mask.nonzero(as_tuple=True)]
                rejected_tds = tds[(~mask).nonzero(as_tuple=True)]
                # Make a lazy stack of accepted rejected. This stack will have shape
                # (1, 2, total_dialog_turns // 2, traj_len)
                tds = lazy_stack([accepted_tds, rejected_tds]).unsqueeze(0) # 0 is accepted, 1 is rejected
                return tds
            return
        elif tensordict.ndim > 2:
            # keep the time dim at the end
            tensordict = tensordict.flatten(0, -2)
        trajs = tensordict.unbind(0)
        # Iterate over the trajectories
        result = []
        for traj in trajs:
            td_out = self._inv_call(traj)
            if td_out is None:
                continue
            result.append(td_out)
        if result:
            return torch.cat(result, 0)
        return

    def _aggregate_rewards(self, reward: torch.Tensor) -> torch.Tensor:
        """Aggregate the rewards across the dialog turns.

        `reward` is expected to be a nested tensor.

        The default implementation is to take the mean of the rewards across the dialog turns.
        """
        # reward = reward.to_padded_tensor(padding=0.0)
        if reward.ndim < 2 or reward.ndim > 3:
            raise ValueError(
                f"Expected reward to be a 2D or 3D tensor, got {reward.ndim}D tensor"
            )
        return reward.mean(dim=-2).squeeze(-1)

class AcceptanceRewardSampler(Sampler):
    """A sampler for acceptance/rejection sampling."""
    def __init__(self, total_dialog_turns: int) -> None:
        super().__init__()
        self.total_dialog_turns = total_dialog_turns
        self._num_accepted_samples = defaultdict(lambda: 0)
        self._num_rejected_samples = defaultdict(lambda: 0)

    def sample(self, storage: Storage, batch_size: int) -> tuple[Any, dict]:
        # samples an index corresponding to a prompt
        prompt_idx = torch.randint(0, len(storage), (batch_size,))

        # Within that prompt, sample an index corresponding to a dialog turn - independently for accepted and rejected
        higher_accept = torch.tensor([self._num_accepted_samples[prompt_idx.item()] for prompt_idx in prompt_idx])
        higher_rej = torch.tensor([self._num_rejected_samples[prompt_idx.item()] for prompt_idx in prompt_idx])

        # equiv to randint with variable upper bound
        accepted_idx = (torch.rand(higher_accept.shape) * higher_accept).floor().long()
        rejected_idx = (torch.rand(higher_rej.shape) * higher_rej).floor().long()

        # Compound the indices
        accepted_idx = torch.stack([prompt_idx, torch.zeros_like(prompt_idx), accepted_idx])
        rejected_idx = torch.stack([prompt_idx, torch.ones_like(prompt_idx), rejected_idx])
        return torch.cat([accepted_idx, rejected_idx]), {}

    def extend(self, index: int | torch.Tensor) -> None:
        print(f'index: {index}')
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        if isinstance(index, list):
            for i in index:
                self.extend(i)
            return
        if not isinstance(index, int):
            raise ValueError(f"Expected an int, got {type(index)=}")
        # Keep track of the accepted and rejected indices
        self._num_accepted_samples[index] = self._num_accepted_samples[index] + self.total_dialog_turns
        self._num_rejected_samples[index] = self._num_rejected_samples[index] + self.total_dialog_turns

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_accepted_samples": self._num_accepted_samples,
            "num_rejected_samples": self._num_rejected_samples,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._num_accepted_samples = state_dict["num_accepted_samples"]
        self._num_rejected_samples = state_dict["num_rejected_samples"]
    def dumps(self) -> str:
        raise NotImplementedError("Not implemented")
    def loads(self, state_dict: str) -> None:
        raise NotImplementedError("Not implemented")
    def _empty(self):
        self.__init__()

class AcceptanceRewardWriter(RoundRobinWriter):
    def __init__(self, total_dialog_turns: int) -> None:
        super().__init__()
        self.total_dialog_turns = total_dialog_turns
        self._num_accepted_samples = defaultdict(lambda: 0)
        self._num_rejected_samples = defaultdict(lambda: 0)

    def add(self, data: Any) -> torch.Tensor | int:
        pass

    def extend(self, data: Any) -> torch.Tensor:
        pass
    
    def state_dict(self) -> dict[str, Any]:
        return {
            "num_accepted_samples": self._num_accepted_samples,
            "num_rejected_samples": self._num_rejected_samples,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._num_accepted_samples = state_dict["num_accepted_samples"]
        self._num_rejected_samples = state_dict["num_rejected_samples"]
    def dumps(self) -> str:
        raise NotImplementedError("Not implemented")
    def loads(self, state_dict: str) -> None:
        raise NotImplementedError("Not implemented")
    def _empty(self):
        self.__init__()
    