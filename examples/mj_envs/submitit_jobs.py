import time
from os import path

import submitit
import torch
from redq import main as main_redq, parser as parser_redq

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="REDQ_log")

# set timeout in min, and partition for running the job
executor.update_parameters(
    timeout_min=2880,
    slurm_partition="scavenge",
    gpus_per_node=8,
    cpus_per_task=94
    # timeout_min = 2880, gpus_per_node = 4, cpus_per_task = 47, mem = 524288,
)
jobs = []
exp_names = []
seed_list = [1]  # 1, 42, 1988]
use_avg_pooling = [True]
shared_mapping = [True]

envs = []

from mj_envs.envs import *
import gym

envs = [
    k
    for k in gym.envs.registration.registry.env_specs.keys()
    if k.startswith("visual") and k != "visual_kitchen-v3"
]

deps = {}
for _shared_mapping in shared_mapping:
    for _use_avg_pooling in use_avg_pooling:
        for seed in seed_list:
            for env in envs:
                use_avg_pooling_str = ["avg_pooling"] if _use_avg_pooling else []
                shared_mapping_str = ["shared_mapping"] if _shared_mapping else []
                exp_name = "-".join(
                    ["SUBMITIT", "8g", env, "seed", str(seed)]
                    + use_avg_pooling_str
                    + shared_mapping_str
                )
                flags = [
                    "--config",
                    f"redq_configs_pixels/generic.txt",
                    "--env_name",
                    env,
                    "--seed",
                    str(seed),
                    "--exp_name",
                    exp_name,
                    "--collector_devices",
                    "cuda:1",
                    "cuda:2",
                    "cuda:3",
                    "cuda:4",
                    "cuda:5",
                    "cuda:6",
                    "cuda:7",
                    "--num_workers",
                    "14",
                    "--env_per_collector",
                    "2",
                    "--recorder_log_keys",
                    "reward",
                    "solved",
                    "--noops",
                    "3",
                    # "--activation",
                    # "tanh",
                    # "cuda:4",
                    # "cuda:5",
                    # "cuda:6",
                    # "cuda:7",
                ]
                if _use_avg_pooling:
                    flags += ["--use_avg_pooling"]
                if _shared_mapping:
                    flags += ["--shared_mapping"]

                config = parser_redq.parse_args(flags)

                job = executor.submit(main_redq, config)
                print(
                    "flags:",
                    flags,
                    # f"\n\ndependency={dep}",
                    "\n\njob id: ",
                    job.job_id,
                    "\n\nexp_name: ",
                    exp_name,
                )  # ID of your job

                # deps[env] = job.job_id
                jobs.append(job)
                exp_names.append(exp_name)
                time.sleep(3)

                while len(jobs) >= 6:
                    print("waiting for job to complete")
                    job = jobs[0]
                    exp_name = exp_names[0]
                    output = job.result()  # waits for completion and returns output
                    try:
                        folder = output[0]
                        torch.save(output[1:], path.join(folder, f"dump_{exp_name}.t"))
                    except:
                        print("failed to save results")
                    jobs = jobs[1:]
                    exp_name = exp_names[1:]

for job, exp_name in zip(jobs, exp_names):
    output = job.result()  # waits for completion and returns output
    folder = output[0]
    torch.save(output[1:], path.join(folder, f"dump_{exp_name}.t"))
