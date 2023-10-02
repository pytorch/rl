# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.envs import Transform


def append_suffix(key, suffix):
    if isinstance(key, str):
        return key + suffix
    return key[:-1] + (append_suffix(key[-1], suffix),)


class DoneTransform(Transform):
    """Expands the 'done' entries (incl. terminated) to match the reward shape.

    Can be appended to a replay buffer or a collector.
    """
    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            tensordict.set(
                ("next", append_suffix(done_key, "_expand")),
                tensordict.get(("next", done_key))
                .unsqueeze(-1)
                .expand(tensordict.get(("next", self.reward_key)).shape),
            )
        return tensordict
