from torch import multiprocessing as mp
import torch
from torch import distributed as dist

from tensordict import TensorDict
from torch import nn


def create_model():
    return nn.Linear(3, 4)


def get_params_and_grad(model):
    params = TensorDict.from_module(model)

    def set_grad(p):
        p.grad = torch.zeros_like(p.data)
        return p

    params.apply(set_grad)
    grad = params.apply(lambda p: p.grad)
    return params, grad


def remote_compute_grad(rank, ):
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=3,
        init_method=f"tcp://localhost:10003",
    )

    print('initiated', rank)

    # create the same model
    model = create_model()
    params, grad = get_params_and_grad(model)

    # pretend we compute something here
    grad.apply_(lambda x: torch.ones_like(x))
    grad.reduce(0)  # send grads to server, operation is SUM

    # receive latest params
    params.irecv(src=0)
    print(f"agent {rank} received params from server")
    print(params['weight'])


def server():
    torch.distributed.init_process_group(
        "gloo",
        rank=0,
        world_size=3,
        init_method=f"tcp://localhost:10003",
    )

    print('initiated 0')

    # create model and set params to 0
    model = create_model()
    model.weight.data.fill_(0.0)
    model.bias.data.fill_(0.0)

    # get params and grad
    params, grad = get_params_and_grad(model)

    # collect gradients from workers
    print(f"server received grads from agents")
    grad.reduce(0, op=dist.ReduceOp.SUM)  # see reduce doc to see what ops are supported
    grad.apply_(lambda x: x / 2)  # average grads
    print(grad['weight'])

    # update params and send updated version to workers
    params.apply_(lambda p, g: p.data.copy_(g), grad)
    params.isend(dst=1)
    params.isend(dst=2)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    p = mp.Process(target=server, args=())
    p.start()

    procs = [p]
    for i in range(2):
        p = mp.Process(target=remote_compute_grad, args=(i + 1,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
