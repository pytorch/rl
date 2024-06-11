# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This script executes some envs across the Gym library with the explicit scope of testing the throughput using the various TorchRL components.

We test:
- gym async envs embedded in a TorchRL's GymEnv wrapper,
- ParallelEnv with regular GymEnv instances,
- Data collector
- Multiprocessed data collectors with parallel envs.

The tests are executed with various number of cpus, and on different devices.

"""
import time

# import myosuite  # noqa: F401
import torch
import tqdm
from torchrl._utils import timeit
from torchrl.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv
from torchrl.envs.libs.gym import gym_backend as gym_bc, set_gym_backend
from torchrl.envs.utils import RandomPolicy

if __name__ == "__main__":
    avail_devices = ("cpu",)
    if torch.cuda.device_count():
        avail_devices = avail_devices + ("cuda:0",)

    for envname in [
        "CartPole-v1",
        "HalfCheetah-v4",
        "myoHandReachRandom-v0",
        "ALE/Breakout-v5",
    ]:
        # the number of collectors won't affect the resources, just impacts how the envs are split in sub-sub-processes
        for num_workers, num_collectors in zip((32, 64, 8, 16), (8, 8, 2, 4)):
            with open(f"{envname}_{num_workers}.txt".replace("/", "-"), "w+") as log:
                if "myo" in envname:
                    gym_backend = "gym"
                else:
                    gym_backend = "gymnasium"

                total_frames = num_workers * 10_000

                # pure gym
                def make(envname=envname, gym_backend=gym_backend):
                    with set_gym_backend(gym_backend):
                        return gym_bc().make(envname)

                with set_gym_backend(gym_backend):
                    env = gym_bc().vector.AsyncVectorEnv(
                        [make for _ in range(num_workers)]
                    )
                env.reset()
                global_step = 0
                times = []
                start = time.time()
                for _ in tqdm.tqdm(range(total_frames // num_workers)):
                    env.step(env.action_space.sample())
                    global_step += num_workers
                env.close()
                log.write(
                    f"pure gym: {num_workers * 10_000 / (time.time() - start): 4.4f} fps\n"
                )
                log.flush()

                # regular parallel env
                for device in avail_devices:

                    def make(envname=envname, gym_backend=gym_backend):
                        with set_gym_backend(gym_backend):
                            return GymEnv(envname, device="cpu")

                    # env_make = EnvCreator(make)
                    penv = ParallelEnv(num_workers, EnvCreator(make), device=device)
                    with torch.inference_mode():
                        # warmup
                        penv.rollout(2)
                        pbar = tqdm.tqdm(total=num_workers * 10_000)
                        t0 = time.time()
                        data = None
                        for _ in range(100):
                            data = penv.rollout(
                                100, break_when_any_done=False, out=data
                            )
                            pbar.update(100 * num_workers)
                    log.write(
                        f"penv {device}: {num_workers * 10_000 / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    penv.close()
                    timeit.print()
                    del penv

                for device in avail_devices:

                    def make(envname=envname, gym_backend=gym_backend):
                        with set_gym_backend(gym_backend):
                            return GymEnv(envname, device="cpu")

                    env_make = EnvCreator(make)
                    # penv = SerialEnv(num_workers, env_make)
                    penv = ParallelEnv(num_workers, env_make, device=device)
                    collector = SyncDataCollector(
                        penv,
                        RandomPolicy(penv.action_spec),
                        frames_per_batch=1024,
                        total_frames=num_workers * 10_000,
                        device=device,
                    )
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    total_frames = 0
                    t0 = time.time()
                    for data in collector:
                        total_frames += data.numel()
                        pbar.update(data.numel())
                        pbar.set_description(
                            f"single collector + torchrl penv: {total_frames / (time.time() - t0): 4.4f} fps"
                        )
                    log.write(
                        f"single collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    collector.shutdown()
                    del collector

                for device in avail_devices:
                    # gym parallel env
                    def make_env(
                        envname=envname,
                        num_workers=num_workers,
                        gym_backend=gym_backend,
                        device=device,
                    ):
                        with set_gym_backend(gym_backend):
                            penv = GymEnv(envname, num_envs=num_workers, device=device)
                        return penv

                    penv = make_env()
                    # warmup
                    penv.rollout(2)
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    t0 = time.time()
                    for _ in range(100):
                        data = penv.rollout(100, break_when_any_done=False)
                        pbar.update(100 * num_workers)
                    log.write(
                        f"gym penv {device}: {num_workers * 10_000 / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    penv.close()
                    del penv

                for device in avail_devices:
                    # async collector
                    # + torchrl parallel env
                    def make_env(envname=envname, gym_backend=gym_backend):
                        with set_gym_backend(gym_backend):
                            return GymEnv(envname, device="cpu")

                    penv = ParallelEnv(
                        num_workers // num_collectors,
                        EnvCreator(make_env),
                        device=device,
                    )
                    collector = MultiaSyncDataCollector(
                        [penv] * num_collectors,
                        policy=RandomPolicy(penv.action_spec),
                        frames_per_batch=1024,
                        total_frames=num_workers * 10_000,
                        device=device,
                    )
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    total_frames = 0
                    for i, data in enumerate(collector):
                        if i == num_collectors:
                            t0 = time.time()
                        if i >= num_collectors:
                            total_frames += data.numel()
                            pbar.update(data.numel())
                            pbar.set_description(
                                f"collector + torchrl penv: {total_frames / (time.time() - t0): 4.4f} fps"
                            )
                    log.write(
                        f"async collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    collector.shutdown()
                    del collector

                for device in avail_devices:
                    # async collector
                    # + gym async env
                    def make_env(
                        envname=envname,
                        num_workers=num_workers,
                        gym_backend=gym_backend,
                    ):
                        with set_gym_backend(gym_backend):
                            penv = GymEnv(envname, num_envs=num_workers, device="cpu")
                        return penv

                    penv = EnvCreator(
                        lambda num_workers=num_workers // num_collectors: make_env(
                            num_workers=num_workers
                        )
                    )
                    collector = MultiaSyncDataCollector(
                        [penv] * num_collectors,
                        policy=RandomPolicy(penv().action_spec),
                        frames_per_batch=1024,
                        total_frames=num_workers * 10_000,
                        num_sub_threads=num_workers // num_collectors,
                        device=device,
                    )
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    total_frames = 0
                    for i, data in enumerate(collector):
                        if i == num_collectors:
                            t0 = time.time()
                        if i >= num_collectors:
                            total_frames += data.numel()
                            pbar.update(data.numel())
                            pbar.set_description(
                                f"{i} collector + gym penv: {total_frames / (time.time() - t0): 4.4f} fps"
                            )
                    log.write(
                        f"async collector + gym penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    collector.shutdown()
                    del collector

                for device in avail_devices:
                    # sync collector
                    # + torchrl parallel env
                    def make_env(envname=envname, gym_backend=gym_backend):
                        with set_gym_backend(gym_backend):
                            return GymEnv(envname, device="cpu")

                    penv = ParallelEnv(
                        num_workers // num_collectors,
                        EnvCreator(make_env),
                        device=device,
                    )
                    collector = MultiSyncDataCollector(
                        [penv] * num_collectors,
                        policy=RandomPolicy(penv.action_spec),
                        frames_per_batch=1024,
                        total_frames=num_workers * 10_000,
                        device=device,
                    )
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    total_frames = 0
                    for i, data in enumerate(collector):
                        if i == num_collectors:
                            t0 = time.time()
                        if i >= num_collectors:
                            total_frames += data.numel()
                            pbar.update(data.numel())
                            pbar.set_description(
                                f"collector + torchrl penv: {total_frames / (time.time() - t0): 4.4f} fps"
                            )
                    log.write(
                        f"sync collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    collector.shutdown()
                    del collector

                for device in avail_devices:
                    # sync collector
                    # + gym async env
                    def make_env(
                        envname=envname,
                        num_workers=num_workers,
                        gym_backend=gym_backend,
                    ):
                        with set_gym_backend(gym_backend):
                            penv = GymEnv(envname, num_envs=num_workers, device="cpu")
                        return penv

                    penv = EnvCreator(
                        lambda num_workers=num_workers // num_collectors: make_env(
                            num_workers=num_workers
                        )
                    )
                    collector = MultiSyncDataCollector(
                        [penv] * num_collectors,
                        policy=RandomPolicy(penv().action_spec),
                        frames_per_batch=1024,
                        total_frames=num_workers * 10_000,
                        num_sub_threads=num_workers // num_collectors,
                        device=device,
                    )
                    pbar = tqdm.tqdm(total=num_workers * 10_000)
                    total_frames = 0
                    for i, data in enumerate(collector):
                        if i == num_collectors:
                            t0 = time.time()
                        if i >= num_collectors:
                            total_frames += data.numel()
                            pbar.update(data.numel())
                            pbar.set_description(
                                f"{i} collector + gym penv: {total_frames / (time.time() - t0): 4.4f} fps"
                            )
                    log.write(
                        f"sync collector + gym penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n"
                    )
                    log.flush()
                    collector.shutdown()
                    del collector
    exit()
