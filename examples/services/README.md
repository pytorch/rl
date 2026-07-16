# TorchRL service examples

These examples run the same TensorDict environment, replay, optimization, and
logging loop with different service placements. Backend selection is confined
to construction; the training loop consumes domain-compatible clients.

## Files

| File | Purpose |
| --- | --- |
| `multi_service_single_process.py` | Direct logger and replay buffer with threaded inference |
| `multi_service_multiprocess.py` | Process logger and inference server with a direct replay buffer |
| `multi_service_ray.py` | Ray logger, replay-buffer, and inference owners |
| `ray_collector_services.py` | Scoped Ray ownership and Gloo transport composed through `Collector` |
| `multi_service_utils.py` | Shared service construction and training loop |
| `distributed_services.py` | Basic service-registry usage |

## Dependencies

Run the examples from a TorchRL source checkout or an environment containing
TorchRL and its dependencies. The multi-service examples use Gymnasium:

```bash
pip install gymnasium
```

The Ray profile additionally requires Ray:

```bash
pip install ray
```

## Run

From the repository root:

```bash
python examples/services/multi_service_single_process.py
python examples/services/multi_service_multiprocess.py
python examples/services/multi_service_ray.py
python examples/services/ray_collector_services.py
```

All entry points accept the same training arguments:

```bash
python examples/services/multi_service_multiprocess.py \
    --steps 64 \
    --batch-size 16 \
    --log-dir /tmp/torchrl-service-example
```

## Shared training loop

`multi_service_utils.py` constructs the owners, obtains their clients, and
runs this backend-neutral loop:

```python
td = env.reset()
for step in range(steps):
    td = policy(td)
    step_td = env.step(td)
    replay_buffer.add(step_td)
    td = env.step_mdp(step_td)

    sample = replay_buffer.sample()
    optimizer.zero_grad()
    loss = loss_fn(sample)
    loss.sum(reduce=True).backward()
    optimizer.step()

    logger.log_scalar("train/loss", float(loss["loss"].detach()), step=step)
```

The loss is a TensorDict, so `sum(reduce=True)` reduces its loss terms to one
scalar tensor before backpropagation.

## Deployment profiles

| Profile | Inference | Logger | Replay buffer | Training loop |
| --- | --- | --- | --- | --- |
| Single process | Background thread | Direct | Direct | Driver |
| Multiprocess | Spawned process | Spawned process | Direct | Driver |
| Ray | Ray actor | Ray actor | Ray actor | Driver |

Replay buffers expose direct and Ray service backends. The multiprocess profile
therefore keeps its replay buffer in the driver while moving inference and
logging into spawned processes.

## Ownership and cleanup

The driver retains each owner and passes only clients into the training loop.
An `ExitStack` shuts consumers down before their services and keeps the logger
alive until final metrics have been flushed. The direct profile uses identity
clients, while process and Ray profiles use restricted clients without
lifecycle methods.
