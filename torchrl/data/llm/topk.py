# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.transforms import Transform


class TopKRewardSelector(Transform):
    """A replay-buffer transform that selects the top-k rewards for each prompt.

    Args:
        total_dialog_turns (int): Number of dialog turns to keep in memory for the top-k selection.
        topk_size (int): Number of top-k rewards to select. Must be smaller than or equal to total_dialog_turns.
        prompt_key (NestedKey): Key to the prompt in the tensordict. Defaults to "text".
        rewards_key (NestedKey): Key to the rewards in the tensordict. Defaults to ("next", "reward").
        done_key (NestedKey): Key to the done state in the tensordict. Defaults to ("next", "done").
        verbose (bool): Whether to print verbose information. Defaults to `False`.

    Example:
        >>> from torchrl.data import ReplayBuffer, LazyStackStorage, SamplerWithoutReplacement
        >>> from tensordict import TensorDict, lazy_stack
        >>> import torch
        >>> from torchrl.data.llm.topk import TopKRewardSelector
        >>> # Create a replay buffer with 50 items, a sampler that samples without replacement, and a batch size of 5
        >>> rb = ReplayBuffer(
        ...     storage=LazyStackStorage(50),
        ...     sampler=SamplerWithoutReplacement,
        ...     batch_size=5,
        ... )
        >>> # Create a tensordict with 50 items, each with 10 dialog turns
        >>> td = lazy_stack(
        ...     [
        ...         TensorDict(
        ...             {
        ...                 ("next", "done"): torch.full((1, 1), True),
        ...                 # Reward for i+5 tokens
        ...                 ("next", "reward"): torch.full((i + 5, 1), i),
        ...                 # total of 10 dialogs per prompt
        ...                 "text": f"Prompt {i // 5}",
        ...             }
        ...         )
        ...         for i in range(50)
        ...     ]
        ... )
        >>> # Create a top-k reward selector with 5 dialog turns and a top-k size of 3
        >>> topk = TopKRewardSelector(total_dialog_turns=5, topk_size=3)
        >>> rb.append_transform(topk)
        >>> for _td in td.chunk(25):
        ...     rb.extend(_td)
        >>> # Only wrote top3 of 50 items in 10 groups of 5
        >>>  assert rb.write_count == 30
        >>> assert len(rb) == 30
        >>> r3 = rb[:3].get(("next", "reward"), as_padded_tensor=True).squeeze()
        >>> # 0 and 1 are missing because they're not part of the top-k
        >>> assert (
        ...     r3 == torch.tensor(
        ...         [
        ...             [4, 4, 4, 4, 4, 4, 4, 4, 4],
        ...             [3, 3, 3, 3, 3, 3, 3, 3, 0],
        ...             [2, 2, 2, 2, 2, 2, 2, 0, 0],
        ...         ]
        ...     )
        ... ).all()
    """

    def __init__(
        self,
        total_dialog_turns: int,
        topk_size: int,
        prompt_key: NestedKey = "text",
        rewards_key: NestedKey = ("next", "reward"),
        done_key: NestedKey = ("next", "done"),
        verbose: bool = True,
    ):
        super().__init__()
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.done_key = done_key
        self.queues = defaultdict(lambda: deque(maxlen=total_dialog_turns))
        self.total_dialog_turns = total_dialog_turns
        self.topk_size = topk_size
        if topk_size > total_dialog_turns:
            raise ValueError(
                f"topk_size must be smaller than or equal to total_dialog_turns, got {topk_size=} and {total_dialog_turns=}"
            )
        self.verbose = verbose

    def forward(self, tensordict: TensorDictBase) -> Any:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Tensordict can be any number of dims, but it must contain entire trajectories
        if tensordict.ndim == 1:
            # Check how many done states we have
            num_done = tensordict[self.done_key].sum()
            if num_done > 1:
                done_idx = tensordict[self.done_key].nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                tensordicts = tensordict.split(splits)
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts) if tensordicts else None
            # Then we have a single trajectory
            if not tensordict[-1][self.done_key].all():
                raise RuntimeError("Expected the trajectory to be done.")
            prompt = tensordict[0][self.prompt_key]
            if not isinstance(prompt, str):
                raise TypeError(f"Expected a string as prompt, got {type(prompt)=}")
            self.queues[prompt].append(tensordict)
            if len(self.queues[prompt]) == self.total_dialog_turns:
                if self.verbose:
                    torchrl_logger.info(f"Getting top-k rewards for {prompt=}")
                # Cat is the most robust way to combine the trajs
                tds = torch.cat(list(self.queues[prompt]), -1)
                # Collect rewards
                reward = tds.get(self.rewards_key, as_nested_tensor=True)
                reward = self._aggregate_rewards(reward)
                # Check if all rewards are equal
                if (reward == reward[0]).all():
                    # If all rewards are equal, we can't select top-k
                    if self.verbose:
                        torchrl_logger.warning(
                            f"All rewards are equal ({reward.unique()=})"
                        )
                    return
                # Filter out rewards below median
                median_reward = reward.median(dim=-1, keepdim=True)[0]
                mask = reward > median_reward
                filtered_reward = reward[mask]
                filtered_indices = mask.nonzero(as_tuple=True)[0]
                # Get top-k from filtered rewards
                topk_reward = filtered_reward.topk(
                    k=min(self.topk_size, len(filtered_indices)), dim=-1
                )
                if not topk_reward.indices.numel():
                    if self.verbose:
                        torchrl_logger.warning(
                            f"No top-{self.topk_size} rewards found ({reward=})"
                        )
                    return
                # Map back to original indices
                selected_indices = filtered_indices[topk_reward.indices]
                tds = tds[selected_indices]
                if self.verbose:
                    torchrl_logger.info(
                        f"Selected top-{self.topk_size} rewards, with reward {topk_reward.values=}"
                    )
                return tds
            return
        elif tensordict.ndim > 2:
            # keep the time dim at the end
            tensordict = tensordict.flatten(0, -2)
        trajs = tensordict.unbind(-1)
        # Iterate over the trajectories
        result = []
        for traj in trajs:
            td_out = self._inv_call(traj)
            if td_out is None:
                continue
            result.append(td_out)
        if result:
            return torch.cat(result, -1)
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
