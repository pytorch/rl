# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from _utils_internal import get_available_devices
from tensordict.tensordict import assert_allclose_td, TensorDict
from torchrl.collectors.utils import split_trajectories
from torchrl.data.postprocs.postprocs import MultiStep


@pytest.mark.parametrize("n", range(13))
@pytest.mark.parametrize("device", get_available_devices())
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
            "next": {key: total_obs[:, 1:] * mask.to(torch.float)},
            "done": done,
            "reward": torch.randn(1, T, 1, device=device).expand(b, T, 1)
            * mask.to(torch.float),
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

    if n == 0:
        assert_allclose_td(tensordict, ms_tensordict.select(*list(tensordict.keys())))

    # assert that done at last step is similar to unterminated traj
    assert (ms_tensordict.get("gamma")[4] == ms_tensordict.get("gamma")[0]).all()
    assert (
        ms_tensordict.get(("next", key))[4] == ms_tensordict.get(("next", key))[0]
    ).all()
    assert (
        ms_tensordict.get("steps_to_next_obs")[4]
        == ms_tensordict.get("steps_to_next_obs")[0]
    ).all()

    # check that next obs is properly replaced, or that it is terminated
    next_obs = ms_tensordict.get(key)[:, (1 + ms.n_steps_max) :]
    true_next_obs = ms_tensordict.get(("next", key))[:, : -(1 + ms.n_steps_max)]
    terminated = ~ms_tensordict.get("nonterminal")
    assert ((next_obs == true_next_obs) | terminated[:, (1 + ms.n_steps_max) :]).all()

    # test gamma computation
    torch.testing.assert_close(
        ms_tensordict.get("gamma"), ms.gamma ** ms_tensordict.get("steps_to_next_obs")
    )

    # test reward
    if n > 0:
        assert (
            ms_tensordict.get("reward") != ms_tensordict.get("original_reward")
        ).any()
    else:
        assert (
            ms_tensordict.get("reward") == ms_tensordict.get("original_reward")
        ).all()


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
        for _ in range(traj_len):
            done = step_count == traj_ids  # traj_id 0 has 0 steps, 1 has 1 step etc.

            td = TensorDict(
                source={
                    ("collector", "traj_ids"): traj_ids,
                    "a": traj_ids.clone().unsqueeze(-1),
                    ("collector", "step_count"): step_count,
                    "workers": workers,
                    "done": done.unsqueeze(-1),
                },
                batch_size=[num_workers],
            )
            out.append(td.clone())
            step_count += 1

            traj_ids[done] = traj_ids.max() + torch.arange(1, done.sum() + 1)
            step_count[done] = 0

        out = torch.stack(out, 1).contiguous()
        return out

    @pytest.mark.parametrize("num_workers", range(3, 34, 3))
    @pytest.mark.parametrize("traj_len", [10, 17, 50, 97])
    def test_splits(self, num_workers, traj_len):

        trajs = TestSplits.create_fake_trajs(num_workers, traj_len)
        assert trajs.shape[0] == num_workers
        assert trajs.shape[1] == traj_len
        split_trajs = split_trajectories(trajs)
        assert (
            split_trajs.shape[0] == split_trajs.get(("collector", "traj_ids")).max() + 1
        )
        assert (
            split_trajs.shape[1]
            == split_trajs.get(("collector", "step_count")).max() + 1
        )

        assert split_trajs.get(("collector", "mask")).sum() == num_workers * traj_len

        assert split_trajs.get("done").sum(1).max() == 1
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
                out_mask.get("done")[idx_traj_id].sum() == 1
            )  # option 1: trajectory is complete
            c2 = out_mask.get("done")[idx_traj_id].sum() == 0
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
