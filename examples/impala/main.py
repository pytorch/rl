import argparse
import os

parser = argparse.ArgumentParser(
    description="Parameter-Server RPC based training")
parser.add_argument(
    "--world_size",
    type=int,
    default=4,
    help="""Total number of participating processes. Should be the sum of
    master node and all training nodes.""")
parser.add_argument(
    "--rank_variable",
    type=str,
    default="SLURM_LOCALID",
    help="Global rank of this process. Pass in 0 for master.")
parser.add_argument(
    "--num_gpus",
    type=int,
    default=0,
    help="""Number of GPUs to use for training,""")
parser.add_argument(
    "--master_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
    Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
    provided. Master must be able to accept network traffic on the host and port.""")

parser.add_argument(
    '--env_config',
    default="examples/impala/configs/atari.yaml",
    type=str,
    help="yaml config file containing the information about the environments to train on."
)

parser.add_argument(
    '--num_procs_per_worker',
    default=32,
    type=int,
    help="number of processes launched on each worker"
)

parser.add_argument(
    '--task',
    default='atari',
    type=str,
    help="Task to be solved. ",
    choices=["atari"]
)

if __name__ == '__main__':

    args = parser.parse_args()
    args.rank = os.environ[args.rank_variable]
    assert args.rank is not None, "must provide rank argument through the --rank_variable flag."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
