# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
from torchrl.record import VideoRecorder


class TestVideoRecorder:
    def test_can_init_with_fps(self):
        recorder = VideoRecorder(None, None, fps=30)

        assert recorder is not None


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
