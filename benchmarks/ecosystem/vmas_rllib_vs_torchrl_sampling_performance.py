# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle

import time
from pathlib import Path

import numpy as np

import ray

import vmas
from matplotlib import pyplot as plt
from ray import tune

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune import register_env
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.vmas import VmasEnv
from vmas import Wrapper


def store_pickled_evaluation(name: str, evaluation: dict):
    save_folder = f"{os.path.dirname(os.path.realpath(__file__))}"
    file = f"{save_folder}/{name}.pkl"

    pickle.dump(evaluation, open(file, "wb"))


def load_pickled_evaluation(
    name: str,
):
    save_folder = f"{os.path.dirname(os.path.realpath(__file__))}"
    file = Path(f"{save_folder}/{name}.pkl")

    if file.is_file():
        return pickle.load(open(file, "rb"))
    return None


def run_vmas_torchrl(
    scenario_name: str, n_envs: int, n_steps: int, device: str, seed: int = 0
):
    env = VmasEnv(
        scenario_name,
        device=device,
        num_envs=n_envs,
        continuous_actions=False,
        seed=seed,
    )

    collector = SyncDataCollector(
        env,
        policy=None,
        device=device,
        frames_per_batch=n_envs * n_steps,
        total_frames=n_envs * n_steps,
    )

    init_time = time.time()

    for _data in collector:
        pass

    total_time = time.time() - init_time
    collector.shutdown()
    return total_time


def run_vmas_rllib(
    scenario_name: str, n_envs: int, n_steps: int, device: str, seed: int = 0
):
    class TimerCallback(DefaultCallbacks):
        result_time = None

        def on_train_result(
            self,
            *,
            algorithm,
            result: dict,
            **kwargs,
        ) -> None:
            TimerCallback.result_time = (
                result["timers"]["training_iteration_time_ms"]
                - result["timers"]["learn_time_ms"]
            )

    def env_creator(config: dict):
        env = vmas.make_env(
            scenario=config["scenario_name"],
            num_envs=config["num_envs"],
            device=config["device"],
            continuous_actions=False,
            wrapper=Wrapper.RLLIB,
        )
        return env

    if not ray.is_initialized():
        ray.init()
    register_env(scenario_name, lambda config: env_creator(config))

    num_gpus = 0.5 if device == "cuda" else 0
    num_gpus_per_worker = 0.5 if device == "cuda" else 0
    tune.run(
        PPOTrainer,
        stop={"training_iteration": 1},
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "train_batch_size": n_envs * n_steps,
            "rollout_fragment_length": n_steps,
            "sgd_minibatch_size": n_envs * n_steps,
            "num_gpus": num_gpus,
            "num_workers": 0,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": n_envs,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": device,
                "num_envs": n_envs,
                "scenario_name": scenario_name,
                "max_steps": n_steps,
            },
            "callbacks": TimerCallback,
        },
    )
    assert TimerCallback.result_time is not None
    TimerCallback.result_time /= 1_000  # convert to seconds
    return TimerCallback.result_time


def run_comparison_torchrl_rllib(
    scenario_name: str,
    device: str,
    n_steps: int = 100,
    max_n_envs: int = 3000,
    step_n_envs: int = 3,
):
    """

    Args:
        scenario_name (str): name of scenario to benchmark
        device (str):  device to ron comparison on ("cpu" or "cuda")
        n_steps (int):  number of environment steps
        max_n_envs (int): the maximum number of parallel environments to test
        step_n_envs (int): the step size in number of environments from 1 to max_n_envs

    """
    list_n_envs = np.linspace(1, max_n_envs, step_n_envs)

    figure_name = f"VMAS_{scenario_name}_{n_steps}_{device}_steps_rllib_vs_torchrl"
    figure_name_pkl = figure_name + f"_range_{1}_{max_n_envs}_num_{step_n_envs}"

    evaluation = load_pickled_evaluation(figure_name_pkl)
    if not evaluation:
        evaluation = {}
    for framework in ["TorchRL", "RLlib"]:
        if framework not in evaluation.keys():
            torchrl_logger.info(f"\nFramework {framework}")
            vmas_times = []
            for n_envs in list_n_envs:
                n_envs = int(n_envs)
                torchrl_logger.info(f"Running {n_envs} environments")
                if framework == "TorchRL":
                    vmas_times.append(
                        (n_envs * n_steps)
                        / run_vmas_torchrl(
                            scenario_name=scenario_name,
                            n_envs=n_envs,
                            n_steps=n_steps,
                            device=device,
                        )
                    )
                else:
                    vmas_times.append(
                        (n_envs * n_steps)
                        / run_vmas_rllib(
                            scenario_name=scenario_name,
                            n_envs=n_envs,
                            n_steps=n_steps,
                            device=device,
                        )
                    )
                torchrl_logger.info(f"fps {vmas_times[-1]}s")
            evaluation[framework] = vmas_times

    store_pickled_evaluation(name=figure_name_pkl, evaluation=evaluation)

    fig, ax = plt.subplots()
    for key, item in evaluation.items():
        ax.plot(
            list_n_envs,
            item,
            label=key,
        )

    plt.xlabel("Number of batched environments", fontsize=14)
    plt.ylabel("Frames per second", fontsize=14)
    ax.legend(loc="upper left")

    ax.set_title(
        f"Execution time of '{scenario_name}' for {n_steps} steps on {device}.",
        fontsize=8,
    )

    save_folder = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(f"{save_folder}/{figure_name}.pdf")


if __name__ == "__main__":
    # pip install matplotlib
    # pip install "ray[rllib]"==2.1.0
    # pip install torchrl
    # pip install vmas
    # pip install numpy==1.23.5

    run_comparison_torchrl_rllib(
        scenario_name="simple_spread",
        device="cuda",
        n_steps=100,
        max_n_envs=30000,
        step_n_envs=10,
    )
