# Flow Q-learning

State-based offline and offline-to-online [FQL](https://arxiv.org/abs/2502.02538),
using TorchRL policies, losses, replay buffers, collectors, and
`OfflineToOnlineTrainer`. The default configuration follows the
[official reference](https://github.com/seohongpark/fql) for
`antmaze-large-navigate-singletask-v0`: four 512-unit GELU layers, critic
LayerNorm after GELU, Xavier initialization, ten Euler steps, and clipped-double-Q
targets. Other tasks may need different `loss.alpha`, `loss.gamma`, and
`loss.q_aggregation` values; the general FQL default aggregation is `mean`.

## Setup and smoke test

Install TorchRL in your development environment, then install the recipe dependencies:

```bash
python -m pip install -r sota-implementations/fql/requirements.txt
bash sota-implementations/fql/run.sh device=cpu \
  dataset.name=null dataset.random_frames=256 env.max_episode_steps=50 \
  optim.offline_steps=100 optim.online_steps=100 optim.batch_size=32 \
  network.width=64 network.depth=2 evaluation.interval=100 evaluation.episodes=2 \
  hydra.run.dir=outputs/fql/smoke
```

The launcher uses the active Python environment. `device` selects the training
device; pass `device=cpu` to run without CUDA.
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
bash sota-implementations/fql/run.sh device=cuda \
  hydra.run.dir=outputs/fql/offline

# Offline to online: append one transition per update.
bash sota-implementations/fql/run.sh device=cuda optim.online_steps=1000000 \
  hydra.run.dir=outputs/fql/online

# Resume the saved configuration, overriding the total update budget.
bash sota-implementations/fql/run.sh resume=outputs/fql/offline/checkpoints \
  optim.offline_steps=2000000
```

`dataset.root` selects the OGBench cache. This recipe supports state-based
single-task OGBench datasets, whose loader supplies rewards and continuation
masks. Actions are normalized to `[-1, 1]`; dataset actions are clipped inside
those bounds by `1e-5`. Dataset episode boundaries and environment truncations
bootstrap; true terminations do not.

The replay buffer has capacity for the offline dataset plus all configured online
steps. Both phases sample uniformly with replacement. Online collection uses the
current stochastic one-step actor with one gradient update per transition.
Teacher integration runs only during training. Target critics use TorchRL's
standard post-optimizer soft update. The official reference uses pre-optimizer
critic parameters, so this ordering differs from the reference.

`optim.compile_loss=true` compiles the loss callable. Evaluation stacks
episode TensorDicts and reduces return and OGBench success over completed
episodes, while preserving the training CPU/CUDA random streams.

## Logging and checkpoints

The default TorchRL CSV logger writes scalar files under the Hydra output
directory's `fql/scalars/` directory. Use `logger.backend=wandb` for Weights &
Biases, `logger.mode=offline` for local W&B logging, or `logger.backend=` to
disable logging and evaluation.

Trainer checkpoints under `checkpoints/` include loss/target weights,
optimizer, replay, random state, collector progress, logger and configuration.
The latest checkpoint is retained. A first SIGINT or SIGTERM finishes the
current update, saves a checkpoint and exits with status 130; a second signal
interrupts immediately. Partial evaluations are discarded.

## Reference comparison

Run both implementations on the same dataset, seeds, update budget and evaluation
schedule. The default protocol uses seeds 0, 1 and 2, one million offline updates,
evaluation every 100,000 updates and 50 evaluation episodes. Add one million
online steps to both runs for the offline-to-online comparison. Keep Q
normalization disabled.

TorchRL command, repeated for each seed:

```bash
bash sota-implementations/fql/run.sh device=cuda seed=0 \
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

Record source revisions, versions, hardware and commands with the learning
curves. Short smoke runs establish execution correctness; learning performance
must be measured separately.

## Trainer API

`FQLTrainer` extends `OfflineToOnlineTrainer` and reuses its replay, optimizer,
target-update and training hooks. `trainer.update()` performs one replay update;
`trainer.train()` runs offline pretraining followed by optional online collection.
Pass a collector factory to create the collector after pretraining.

The Trainer accepts TorchRL loggers and native checkpoints. Structured Hydra
configurations are registered as `trainer/fql` and `loss/fql`. Optimizer and
target-updater configurations are partials bound to the shared loss instance.
The recipe's `matmul_precision` option defaults to `high`.
