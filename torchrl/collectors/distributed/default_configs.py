import os

TCP_PORT = os.environ.get("TCP_PORT", "10003")
IDLE_TIMEOUT = os.environ.get("RCP_IDLE_TIMEOUT", 10)

MAX_TIME_TO_CONNECT = 1000

SLEEP_INTERVAL = 1e-6

DEFAULT_SLURM_CONF = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 0,
}  #: Default value of the SLURM jobs

DEFAULT_SLURM_CONF_MAIN = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 1,
}  #: Default value of the SLURM main job

DEFAULT_TENSORPIPE_OPTIONS = {
    "num_worker_threads": 16,
    "rpc_timeout": 10_000,
    "_transports": ["uv"],
}
