# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/train.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker
```

The Walker preset matches the 640,867 trainable parameters of the reference
implementation's `size1m` configuration, uses 16 environments, batches of 16
sequences of length 64, a replay ratio of 1024, and 1.1 million environment
steps. BF16 training is enabled on CUDA. It logs stochastic training-episode
returns against environment steps, matching the reference curve protocol
without relying on wall-clock-dependent training iterations.

The Walker task is seeded from `env.seed`, as every other TorchRL example is;
pass `env.use_seed=false` for the reference's unseeded DMC resets. The step
axis counts initial and reset-only driver records as the reference
implementation does.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

For a three-seed median and interquartile reproduction run:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir dmc_walker_runs
```

The benchmark writes one metrics file per seed plus `summary.json`, aggregates
the stochastic training returns into median and interquartile curves over fixed
windows, and fails when the final window median falls short. The seeds, the
window and the threshold come from the `benchmark` block of
`config_dmc_walker.yaml`, which ships three seeds, 50,000-step windows and a
minimum final median return of 900; `benchmark.*` Hydra overrides change them,
as in `benchmark.seeds=[0,1,2,3,4]`. `env.seed` and `logger.metrics_jsonl` are
set per run and are rejected as overrides, since either would collapse the
seeds onto one trajectory. Full learning-curve runs are intended for scheduled
or manual validation; pull-request CI uses short smoke overrides.

For a smaller ablation, shorten the run rather than the window:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir smoke \
  collector.total_frames=100000 \
  benchmark.minimum_final_median_return=0
```

Every worker runs to the same time limit, so episodes finish in bursts one
episode apart: `(env.max_episode_steps + 1) * collector.num_envs`, or 16,016
records for the preset. A window narrower than that holds no completed episode
over most of the run, so the script refuses one before launching anything. The
command above keeps the 50,000-step window and still fills two of them with
about 48 episodes each.
