# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari game data collection benchmark with stable-baselines3
===========================================================

Runs an Atari game with a random policy using a multiprocess async data collector.

Image size: torch.Size([210, 160, 3])

Performance results with default configuration:
+-------------------------------+--------------------------------------------------+
| Machine specs                 |  3x A100 GPUs,                                   |
|                               | Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz   |
|                               |                                                  |
+===============================+==================================================+
|                               | 1176.7944 fps                                    |
+-------------------------------+--------------------------------------------------+

"""

import time

import tqdm
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
n_envs = 32
env = make_atari_env("PongNoFrameskip-v4", n_envs=n_envs, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)

frames = 0
total_frames = 100_000
pbar = tqdm.tqdm(total=total_frames)
obs = env.reset()
action = None

i = 0
while True:
    if i == 10:
        t0 = time.time()
    elif i >= 10:
        frames += n_envs
    pbar.update(n_envs)
    if action is None:
        action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if frames > total_frames:
        break
    i += 1
t = frames / (time.time() - t0)
print(f"fps: {t}")
