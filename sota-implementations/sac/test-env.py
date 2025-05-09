# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.cuda
import numpy
from utils import make_environment, make_sac_agent
from torchrl.envs import set_exploration_type
from tensordict import assert_close
import hydra

@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg):  # noqa: F821
    rs = []
    for worker_device in (None, "cpu", "cuda:0", "cuda:1"):
        rs.append([])
        for device in (None, "cpu", "cuda:0", "cuda:1"):
            torch.cuda.manual_seed(0)
            torch.manual_seed(0)
            numpy.random.seed(0)
            train_env, eval_env = make_environment(cfg, device=device, worker_device=worker_device)
            train_env.set_seed(0)
            model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)
            with set_exploration_type("RANDOM"):
                r = train_env.rollout(1000, exploration_policy, break_when_any_done=False)
            rs[-1].append(r)
        rs[-1] = torch.stack(rs[-1], dim=0)
    rs = torch.stack(rs, dim=0)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert_close(rs[0, 0], rs[i, j])

if __name__ == "__main__":
    main()
