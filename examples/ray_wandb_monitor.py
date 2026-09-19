from __future__ import annotations

import ray
from torchrl.collectors.distributed import RayCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.monitoring import Every, LoggerMonitor
from torchrl.record.loggers.wandb import WandbLogger


def main():
    # Initialize WandB Logger
    # For a real run, ensure you have wandb installed and are logged in.
    logger = WandbLogger(exp_name="ray_monitor_test", project="torchrl_monitoring")

    # Define the environment constructor
    def env_maker():
        return GymEnv("Pendulum-v1")

    # Set up the Ray Collector with 2 remote workers
    collector = RayCollector(
        [env_maker, env_maker],
        policy=RandomPolicy(action_spec=GymEnv("Pendulum-v1").action_spec),
        frames_per_batch=100,
        total_frames=1000,
    )

    # Monitor the collector and report per-worker stats
    # Setting workers="both" requests both the aggregate and the per-worker metrics
    with LoggerMonitor(logger, poll_interval=0.5) as monitor:
        monitor.watch(
            collector,
            name="ray_collector",
            schedule=Every.counter("frames", 100),
            stats_kwargs={"workers": "both"},
        )

        # Run the collection loop
        for _i, _batch in enumerate(collector):
            # In a real training script, you would do policy updates here.
            pass

    # The LoggerMonitor stops polling and logs final stats on exit
    print("Collection finished. View results in the WandB dashboard!")


if __name__ == "__main__":
    # Initialize ray in local mode or connect to a cluster
    ray.init()
    main()
