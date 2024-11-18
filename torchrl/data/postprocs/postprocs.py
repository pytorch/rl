# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict.utils import expand_right
from torch import nn


def _get_reward(
    gamma: float,
    reward: torch.Tensor,
    done: torch.Tensor,
    max_steps: int,
):
    """Sums the rewards up to max_steps in the future with a gamma decay.

    Supports multiple consecutive trajectories.

    Assumes that the time dimension is the *last* dim of reward and done.
    """
    filt = torch.tensor(
        [gamma**i for i in range(max_steps + 1)],
        device=reward.device,
        dtype=reward.dtype,
    ).view(1, 1, -1)
    # make one done mask per trajectory
    done_cumsum = done.cumsum(-1)
    done_cumsum = torch.cat(
        [torch.zeros_like(done_cumsum[..., :1]), done_cumsum[..., :-1]], -1
    )
    num_traj = done_cumsum.max().item() + 1
    done_cumsum = done_cumsum.expand(num_traj, *done.shape)
    traj_ids = done_cumsum == torch.arange(
        num_traj, device=done.device, dtype=done_cumsum.dtype
    ).view(num_traj, *[1 for _ in range(done_cumsum.ndim - 1)])
    # an expanded reward tensor where each index along dim 0 is a different trajectory
    # Note: rewards could have a different shape than done (e.g. multi-agent with a single
    # done per group).
    # we assume that reward has the same leading dimension as done.
    if reward.shape != traj_ids.shape[1:]:
        # We'll expand the ids on the right first
        traj_ids_expand = expand_right(traj_ids, (num_traj, *reward.shape))
        reward_traj = traj_ids_expand * reward
        # we must make sure that the last dimension of the reward is the time
        reward_traj = reward_traj.transpose(-1, traj_ids.ndim - 1)
    else:
        # simpler use case: reward shape and traj_ids match
        reward_traj = traj_ids * reward

    reward_traj = torch.nn.functional.pad(reward_traj, [0, max_steps], value=0.0)
    shape = reward_traj.shape[:-1]
    if len(shape) > 1:
        reward_traj = reward_traj.flatten(0, reward_traj.ndim - 2)
    reward_traj = reward_traj.unsqueeze(-2)
    summed_rewards = torch.conv1d(reward_traj, filt)
    summed_rewards = summed_rewards.squeeze(-2)
    if len(shape) > 1:
        summed_rewards = summed_rewards.unflatten(0, shape)
    # let's check that our summed rewards have the right size
    if reward.shape != traj_ids.shape[1:]:
        summed_rewards = summed_rewards.transpose(-1, traj_ids.ndim - 1)
        summed_rewards = (summed_rewards * traj_ids_expand).sum(0)
    else:
        summed_rewards = (summed_rewards * traj_ids).sum(0)

    # time_to_obs is the tensor of the time delta to the next obs
    # 0 = take the next obs (ie do nothing)
    # 1 = take the obs after the next
    time_to_obs = (
        traj_ids.flip(-1).cumsum(-1).clamp_max(max_steps + 1).flip(-1) * traj_ids
    )
    time_to_obs = time_to_obs.sum(0)
    time_to_obs = time_to_obs - 1
    return summed_rewards, time_to_obs


class MultiStep(nn.Module):
    """Multistep reward transform.

    Presented in

    | Sutton, R. S. 1988. Learning to predict by the methods of temporal differences. Machine learning 3(1):9–44.

    This module maps the "next" observation to the t + n "next" observation.
    It is an identity transform whenever :attr:`n_steps` is 0.

    Args:
        gamma (:obj:`float`): Discount factor for return computation
        n_steps (integer): maximum look-ahead steps.

    .. note:: This class is meant to be used within a ``DataCollector``.
        It will only treat the data passed to it at the end of a collection,
        and ignore data preceding that collection or coming in the next batch.
        As such, results on the last steps of the batch may likely be biased
        by the early truncation of the trajectory.
        To mitigate this effect, please use :class:`~torchrl.envs.transforms.MultiStepTransform`
        within the replay buffer instead.

    Examples:
        >>> from torchrl.collectors import SyncDataCollector, RandomPolicy
        >>> from torchrl.data.postprocs import MultiStep
        >>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
        >>> env.set_seed(0)
        >>> collector = SyncDataCollector(env, policy=RandomPolicy(env.action_spec),
        ...     frames_per_batch=10, total_frames=2000, postproc=MultiStep(n_steps=4, gamma=0.99))
        >>> for data in collector:
        ...     break
        >>> print(data["step_count"])
        tensor([[0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]])
        >>> # the next step count is shifted by 3 steps in the future
        >>> print(data["next", "step_count"])
        tensor([[ 5],
                [ 6],
                [ 7],
                [ 8],
                [ 9],
                [10],
                [10],
                [10],
                [10],
                [10]])

    """

    def __init__(
        self,
        gamma: float,
        n_steps: int,
    ):
        super().__init__()
        if n_steps <= 0:
            raise ValueError("n_steps must be a non-negative integer.")
        if not (gamma > 0 and gamma <= 1):
            raise ValueError(f"got out-of-bounds gamma decay: gamma={gamma}")

        self.gamma = gamma
        self.n_steps = n_steps
        self.register_buffer(
            "gammas",
            torch.tensor(
                [gamma**i for i in range(n_steps + 1)],
                dtype=torch.float,
            ).reshape(1, 1, -1),
        )
        self.done_key = "done"
        self.done_keys = ("done", "terminated", "truncated")
        self.reward_keys = ("reward",)
        self.mask_key = ("collector", "mask")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Re-writes a tensordict following the multi-step transform.

        Args:
            tensordict: :class:`tensordict.TensorDictBase` instance with
                ``[*Batch x Time-steps] shape.
                The TensorDict must contain a ``("next", "reward")`` and
                ``("next", "done")`` keys.
                All keys that are contained within the "next" nested tensordict
                will be shifted by (at most) :attr:`~.n_steps` frames.
                The TensorDict will also be updated with new key-value pairs:

                - gamma: indicating the discount to be used for the next
                  reward;
                - nonterminal: boolean value indicating whether a step is
                  non-terminal (not done or not last of trajectory);
                - original_reward: previous reward collected in the
                  environment (i.e. before multi-step);
                - The "reward" values will be replaced by the newly computed
                  rewards.
                The ``"done"`` key can have either the shape of the tensordict
                OR the shape of the tensordict followed by a singleton
                dimension OR the shape of the tensordict followed by other
                dimensions. In the latter case, the tensordict *must* be
                compatible with a reshape that follows the done shape (ie. the
                leading dimensions of every tensor it contains must match the
                shape of the ``"done"`` entry).
                The ``"reward"`` tensor can have either the shape of the
                tensordict (or done state) or this shape followed by a singleton
                dimension.

        Returns:
            in-place transformation of the input tensordict.

        """
        return _multi_step_func(
            tensordict,
            done_key=self.done_key,
            done_keys=self.done_keys,
            reward_keys=self.reward_keys,
            mask_key=self.mask_key,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )


def _multi_step_func(
    tensordict,
    *,
    done_key,
    done_keys,
    reward_keys,
    mask_key,
    n_steps,
    gamma,
):
    # in accordance with common understanding of what n_steps should be
    n_steps = n_steps - 1
    tensordict = tensordict.clone(False)
    done = tensordict.get(("next", done_key))

    # we'll be using the done states to index the tensordict.
    # if the shapes don't match we're in trouble.
    ndim = tensordict.ndim
    if done.shape != tensordict.shape:
        if done.shape[-1] == 1 and done.shape[:-1] == tensordict.shape:
            done = done.squeeze(-1)
        else:
            try:
                # let's try to reshape the tensordict
                tensordict.batch_size = done.shape
                tensordict = tensordict.transpose(ndim - 1, tensordict.ndim - 1)
                done = tensordict.get(("next", done_key))
            except Exception as err:
                raise RuntimeError(
                    "tensordict shape must be compatible with the done's shape "
                    "(trailing singleton dimension excluded)."
                ) from err

    if mask_key is not None:
        mask = tensordict.get(mask_key, None)
    else:
        mask = None

    *batch, T = tensordict.batch_size

    summed_rewards = []
    for reward_key in reward_keys:
        reward = tensordict.get(("next", reward_key))

        # sum rewards
        summed_reward, time_to_obs = _get_reward(gamma, reward, done, n_steps)
        summed_rewards.append(summed_reward)

    idx_to_gather = torch.arange(
        T, device=time_to_obs.device, dtype=time_to_obs.dtype
    ).expand(*batch, T)
    idx_to_gather = idx_to_gather + time_to_obs

    # idx_to_gather looks like  tensor([[ 2,  3,  4,  5,  5,  5,  8,  9, 10, 10, 10]])
    # with a done state         tensor([[ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1]])
    # meaning that the first obs will be replaced by the third, the second by the fourth etc.
    # The fifth remains the fifth as it is terminal
    tensordict_gather = (
        tensordict.get("next")
        .exclude(*reward_keys, *done_keys)
        .gather(-1, idx_to_gather)
    )

    tensordict.set("steps_to_next_obs", time_to_obs + 1)
    for reward_key, summed_reward in zip(reward_keys, summed_rewards):
        tensordict.rename_key_(("next", reward_key), ("next", "original_reward"))
        tensordict.set(("next", reward_key), summed_reward)

    tensordict.get("next").update(tensordict_gather)
    tensordict.set("gamma", gamma ** (time_to_obs + 1))
    nonterminal = time_to_obs != 0
    if mask is not None:
        mask = mask.view(*batch, T)
        nonterminal[~mask] = False
    tensordict.set("nonterminal", nonterminal)
    if tensordict.ndim != ndim:
        tensordict = tensordict.apply(
            lambda x: x.transpose(ndim - 1, tensordict.ndim - 1),
            batch_size=done.transpose(ndim - 1, tensordict.ndim - 1).shape,
        )
        tensordict.batch_size = tensordict.batch_size[:ndim]
    return tensordict
