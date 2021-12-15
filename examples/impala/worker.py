import math

import yaml

from atari_env import make_parallel_env as make_parallel_env_atari


def _load_env_names(args):
    with open(args.env_config, 'r') as f:
        env_configs = yaml.safe_load(f)
    return ['-'.join([
        _env,
        env_configs['version']]) for _env in env_configs['envs']]


def repeat_env_names(list_to_rep, n_rep):
    out_list = []
    for i, item in enumerate(list_to_rep):
        out_list += [item for _ in range(n_rep)]
    return out_list


def _get_env_fn(worker_id, world_size, args):
    world_size = world_size - 1
    worker_id = worker_id - 1

    num_procs = args.num_procs_per_worker
    env_names = _load_env_names(args)
    total_procs = num_procs * world_size
    if total_procs >= len(env_names):
        # if there are more processes than envs to run, each worker will have multiple copies of some envs
        proc_per_env = total_procs // len(env_names)
        env_names = repeat_env_names(env_names, proc_per_env)
    else:
        raise Exception("there are more envs to run than processes, consider decreasing number of environments or"
                        "increasing resources")
    n_env_per_worker = int(math.ceil(len(env_names) / world_size))
    idx = range(worker_id * n_env_per_worker, min(len(env_names), (worker_id + 1) * n_env_per_worker))
    env_names = [env_names[i] for i in idx]
    print(f"worker {worker_id} running envs: {env_names} (len={len(env_names)})\n\n")
    num_procs = len(env_names)

    if args.task == 'atari':
        create_env_fn = lambda: make_parallel_env_atari(env_names, n_processes=num_procs)
    else:
        raise NotImplementedError(f"task {args.task} not yet supported")
    return create_env_fn


def run_worker(worker_id, world_size, args):
    create_env_fn = _get_env_fn(worker_id, world_size, args)


if __name__ == "__main__":
    from main import parser

    args = parser.parse_args(["--num_gpus", "0", "--rank_variable", "RANK"])
    total = 0
    for i in range(1, 10):
        fn = _get_env_fn(i, 10, args)
        env = fn()
        env.reset()
        env.rand_step()
        total += env.num_workers
        env.close()
    print(f'total: {total}')
