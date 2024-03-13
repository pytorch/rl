# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools

import pytest
import torch
from _utils_internal import get_default_devices
from tensordict import assert_allclose_td, TensorDict
from torchrl.collectors.utils import split_trajectories
from torchrl.data.postprocs.postprocs import MultiStep


@pytest.mark.parametrize("n", range(1, 14))
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("key", ["observation", "pixels", "observation_whatever"])
def test_multistep(n, key, device, T=11):
    torch.manual_seed(0)

    # mock data
    b = 5

    done = torch.zeros(b, T, 1, dtype=torch.bool, device=device)
    done[0, -1] = True
    done[1, -2] = True
    done[2, -3] = True
    done[3, -4] = True

    terminal = done.clone()
    terminal[:, -1] = done.sum(1) != 1

    mask = done.clone().cumsum(1).cumsum(1) >= 2
    mask = ~mask

    total_obs = torch.randn(1, T + 1, 1, device=device).expand(b, T + 1, 1)
    tensordict = TensorDict(
        source={
            key: total_obs[:, :T] * mask.to(torch.float),
            "done": done,
            "next": {
                key: total_obs[:, 1:] * mask.to(torch.float),
                "done": done,
                "reward": torch.randn(1, T, 1, device=device).expand(b, T, 1)
                * mask.to(torch.float),
            },
            "collector": {"mask": mask},
        },
        batch_size=(b, T),
    ).to(device)

    ms = MultiStep(
        0.9,
        n,
    ).to(device)
    ms_tensordict = ms(tensordict.clone())

    assert ms_tensordict.get("done").max() == 1

    if n == 1:
        assert_allclose_td(
            tensordict, ms_tensordict.select(*list(tensordict.keys(True, True)))
        )

    # assert that done at last step is similar to unterminated traj
    torch.testing.assert_close(
        ms_tensordict.get("gamma")[4], ms_tensordict.get("gamma")[0]
    )
    torch.testing.assert_close(
        ms_tensordict.get(("next", key))[4], ms_tensordict.get(("next", key))[0]
    )
    torch.testing.assert_close(
        ms_tensordict.get("steps_to_next_obs")[4],
        ms_tensordict.get("steps_to_next_obs")[0],
    )

    # check that next obs is properly replaced, or that it is terminated
    next_obs = ms_tensordict.get(key)[:, (ms.n_steps) :]
    true_next_obs = ms_tensordict.get(("next", key))[:, : -(ms.n_steps)]
    terminated = ~ms_tensordict.get("nonterminal")
    assert ((next_obs == true_next_obs).all(-1) | terminated[:, (ms.n_steps) :]).all()

    # test gamma computation
    torch.testing.assert_close(
        ms_tensordict.get("gamma"), ms.gamma ** ms_tensordict.get("steps_to_next_obs")
    )

    # test reward
    if n > 1:
        assert (
            ms_tensordict.get(("next", "reward"))
            != ms_tensordict.get(("next", "original_reward"))
        ).any()
    else:
        torch.testing.assert_close(
            ms_tensordict.get(("next", "reward")),
            ms_tensordict.get(("next", "original_reward")),
        )


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize(
    "batch_size",
    [
        [4],
        [],
        [1],
        [2, 3],
    ],
)
@pytest.mark.parametrize("T", [10, 1, 2])
@pytest.mark.parametrize("obs_dim", [[1], []])
@pytest.mark.parametrize("unsq_reward", [True, False])
@pytest.mark.parametrize("last_done", [True, False])
@pytest.mark.parametrize("n_steps", [4, 2, 1])
def test_mutistep_cattrajs(
    batch_size, T, obs_dim, unsq_reward, last_done, device, n_steps
):
    # tests multi-step in the presence of consecutive trajectories.
    obs = torch.randn(*batch_size, T + 1, *obs_dim)
    reward = torch.rand(*batch_size, T)
    action = torch.rand(*batch_size, T)
    done = torch.zeros(*batch_size, T + 1, dtype=torch.bool)
    done[..., T // 2] = 1
    if last_done:
        done[..., -1] = 1
    if unsq_reward:
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

    td = TensorDict(
        {
            "obs": obs[..., :-1] if not obs_dim else obs[..., :-1, :],
            "action": action,
            "done": done[..., :-1] if not unsq_reward else done[..., :-1, :],
            "next": {
                "obs": obs[..., 1:] if not obs_dim else obs[..., 1:, :],
                "done": done[..., 1:] if not unsq_reward else done[..., 1:, :],
                "reward": reward,
            },
        },
        batch_size=[*batch_size, T],
        device=device,
    )
    ms = MultiStep(0.98, n_steps)
    tdm = ms(td)
    if n_steps == 1:
        # n_steps = 0 has no effect
        for k in td["next"].keys():
            assert (tdm["next", k] == td["next", k]).all()
    else:
        next_obs = []
        obs = td["next", "obs"]
        done = td["next", "done"]
        if obs_dim:
            obs = obs.squeeze(-1)
        if unsq_reward:
            done = done.squeeze(-1)
        for t in range(T):
            idx = t + n_steps - 1
            while (done[..., t:idx].any() and idx > t) or idx > done.shape[-1] - 1:
                idx = idx - 1
            next_obs.append(obs[..., idx])
        true_next_obs = tdm.get(("next", "obs"))
        if obs_dim:
            true_next_obs = true_next_obs.squeeze(-1)
        next_obs = torch.stack(next_obs, -1)
        assert (next_obs == true_next_obs).all()


@pytest.mark.parametrize("unsq_reward", [True, False])
def test_unusual_done(unsq_reward):
    batch_size = [10, 3]
    T = 10
    obs_dim = [
        1,
    ]
    last_done = True
    device = torch.device("cpu")
    n_steps = 3

    obs = torch.randn(*batch_size, T + 1, 5, *obs_dim)
    reward = torch.rand(*batch_size, T, 5)
    action = torch.rand(*batch_size, T, 5)
    done = torch.zeros(*batch_size, T + 1, 5, dtype=torch.bool)
    done[..., T // 2, :] = 1
    if last_done:
        done[..., -1, :] = 1
    if unsq_reward:
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

    td = TensorDict(
        {
            "obs": obs[..., :-1, :] if not obs_dim else obs[..., :-1, :, :],
            "action": action,
            "done": done[..., :-1, :] if not unsq_reward else done[..., :-1, :, :],
            "next": {
                "obs": obs[..., 1:, :] if not obs_dim else obs[..., 1:, :, :],
                "done": done[..., 1:, :] if not unsq_reward else done[..., 1:, :, :],
                "reward": reward,
            },
        },
        batch_size=[*batch_size, T],
        device=device,
    )
    ms = MultiStep(0.98, n_steps)
    if unsq_reward:
        with pytest.raises(RuntimeError, match="tensordict shape must be compatible"):
            _ = ms(td)
    else:
        # we just check that it runs
        _ = ms(td)


class TestSplits:
    """Tests the splitting of collected tensordicts in trajectories."""

    @staticmethod
    def create_fake_trajs(
        num_workers=32,
        traj_len=200,
    ):
        traj_ids = torch.arange(num_workers)
        step_count = torch.zeros(num_workers)
        workers = torch.arange(num_workers)

        out = []
        done0 = torch.zeros(num_workers, 1, dtype=torch.bool)
        for _ in range(traj_len):
            done = step_count == traj_ids  # traj_id 0 has 0 steps, 1 has 1 step etc.
            done = done.unsqueeze(-1)
            td = TensorDict(
                source={
                    ("collector", "traj_ids"): traj_ids,
                    "a": traj_ids.clone().unsqueeze(-1),
                    "step_count": step_count,
                    "workers": workers,
                    "done": done0,
                    ("next", "done"): done,
                },
                batch_size=[num_workers],
            )
            done0 = done
            out.append(td.clone())
            step_count += 1

            traj_ids[done.squeeze(-1)] = traj_ids.max() + torch.arange(
                1, done.sum() + 1
            )
            step_count[done.squeeze(-1)] = 0

        out = torch.stack(out, 1).contiguous()
        return out

    @pytest.mark.parametrize("num_workers", range(3, 34, 3))
    @pytest.mark.parametrize("traj_len", [10, 17, 50, 97])
    @pytest.mark.parametrize(
        "constr",
        [
            functools.partial(split_trajectories, prefix="collector"),
            functools.partial(split_trajectories),
            functools.partial(
                split_trajectories, trajectory_key=("collector", "traj_ids")
            ),
        ],
    )
    def test_splits(self, num_workers, traj_len, constr):

        trajs = TestSplits.create_fake_trajs(num_workers, traj_len)
        assert trajs.shape[0] == num_workers
        assert trajs.shape[1] == traj_len
        split_trajs = constr(trajs)
        assert (
            split_trajs.shape[0] == split_trajs.get(("collector", "traj_ids")).max() + 1
        )
        assert split_trajs.shape[1] == split_trajs.get("step_count").max() + 1

        assert split_trajs.get(("collector", "mask")).sum() == num_workers * traj_len

        assert split_trajs.get(("next", "done")).sum(1).max() == 1
        out_mask = split_trajs[split_trajs.get(("collector", "mask"))]
        for i in range(split_trajs.shape[0]):
            traj_id_split = split_trajs[i].get(("collector", "traj_ids"))[
                split_trajs[i].get(("collector", "mask"))
            ]
            assert 1 == len(traj_id_split.unique())

        for w in range(num_workers):
            assert (out_mask.get("workers") == w).sum() == traj_len
        # Assert that either the chain is not done XOR if it is it must have the desired length (equal to traj id by design)
        for i in range(split_trajs.get(("collector", "traj_ids")).max()):
            idx_traj_id = out_mask.get(("collector", "traj_ids")) == i
            # (!=) == (xor)
            c1 = (idx_traj_id.sum() - 1 == i) and (
                out_mask.get(("next", "done"))[idx_traj_id].sum() == 1
            )  # option 1: trajectory is complete
            c2 = out_mask.get(("next", "done"))[idx_traj_id].sum() == 0
            assert c1 != c2, (
                f"traj_len={traj_len}, "
                f"num_workers={num_workers}, "
                f"traj_id={i}, "
                f"idx_traj_id.sum()={idx_traj_id.sum()}, "
                f"done={out_mask.get('done')[idx_traj_id].sum()}"
            )

        assert (
            split_trajs.get(("collector", "traj_ids")).unique().numel()
            == split_trajs.get(("collector", "traj_ids")).max() + 1
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
