# Flow Q-learning

State-based offline and offline-to-online [FQL](https://arxiv.org/abs/2502.02538),
using TorchRL policies, losses, replay buffers, collectors, and target updates.
The default configuration follows the [official reference](https://github.com/seohongpark/fql)
for `antmaze-large-navigate-singletask-v0`: four 512-unit GELU layers, critic
LayerNorm after GELU, Xavier initialization, ten Euler steps, and clipped-double-Q
targets. Other tasks may need different `loss.alpha`, `loss.gamma`, and
`loss.q_aggregation` values; the general FQL default aggregation is `mean`.

## Setup and smoke test

On a Linux A100 (40 GB) host with Python 3, a C++ compiler, NVIDIA drivers, and EGL:

```bash
bash sota-implementations/fql/setup.sh
bash sota-implementations/fql/run.sh \
  dataset.name=null dataset.random_frames=256 env.max_episode_steps=50 \
  optim.offline_steps=100 optim.online_steps=100 optim.batch_size=32 \
  network.width=64 network.depth=2 evaluation.interval=100 evaluation.episodes=2 \
  hydra.run.dir=outputs/fql/smoke
```

Setup creates a local Python 3.11 environment, selects a PyTorch CUDA wheel for
the installed driver, installs TorchRL in editable mode, and checks CUDA access.
The launcher sets headless MuJoCo and noninteractive logging defaults. Metrics
are local JSON lines; no logging account or login is required. Package versions
are saved in `outputs/fql/environment.txt`.

For an existing CPU development environment, run `python
sota-implementations/fql/fql.py device=cpu` with the same smoke overrides.
`dataset.name=null` collects random Pendulum transitions and checks the pipeline;
it does not measure offline learning performance. CI exercises this path through
the existing SOTA smoke harness:

```bash
COMPOSITE_LP_AGGREGATE=0 pytest -q \
  .github/unittest/linux_sota/scripts/test_sota.py -k fql
```

## Training

```bash
# Offline: OGBench downloads the training and validation data on first use.
bash sota-implementations/fql/run.sh hydra.run.dir=outputs/fql/offline

# Offline to online: retain all offline data and append one transition per update.
bash sota-implementations/fql/run.sh optim.online_steps=1000000 \
  hydra.run.dir=outputs/fql/online
```

`dataset.root` selects the OGBench cache. This recipe supports state-based
single-task OGBench datasets, whose loader supplies rewards and continuation
masks. Actions are normalized to `[-1, 1]`; dataset actions are clipped inside
those bounds by `1e-5`. Dataset episode boundaries and environment truncations
bootstrap; true terminations do not.

The replay buffer has capacity for the offline dataset plus all configured online
steps. Both phases sample uniformly with replacement. Online collection uses the
current stochastic one-step actor with one gradient update per transition.
Teacher integration runs only during training. Target critics update before the
optimizer step to match the reference's use of the previous critic parameters.

Each Hydra output directory contains the resolved configuration, `metrics.jsonl`,
and a final `checkpoint.pt` with loss-module weights, optimizer state, configuration,
and step count. Evaluation logs episode return and OGBench success rate. The
checkpoint is for inspection and policy loading; this script does not implement
training resumption or save replay contents.

## Reference comparison

Run both implementations on the same dataset, seeds, and update budget. For the
default task, use seeds 0, 1, and 2, one million offline updates, evaluation every
100,000 updates, and 50 evaluation episodes. Add one million online steps to both
runs for the offline-to-online comparison. Keep Q normalization disabled.

TorchRL command, repeated for each seed:

```bash
bash sota-implementations/fql/run.sh seed=0 \
  hydra.run.dir=outputs/fql/antmaze-seed0
```

In a separate environment containing the official reference and its dependencies:

```bash
WANDB_MODE=disabled python main.py --seed=0 \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --offline_steps=1000000 --online_steps=0 \
  --eval_interval=100000 --eval_episodes=50 \
  --agent.alpha=10 --agent.discount=0.99 --agent.q_agg=min
```

Record both commit hashes, package versions, GPU model, seed, commands, elapsed
time, returns, and success rates. Compare learning curves and mean/spread across
seeds. Short smoke runs establish execution correctness; benchmark parity must
be measured separately.
