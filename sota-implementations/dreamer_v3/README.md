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

The Walker preset uses the 1M-parameter RSSM dimensions, batches of 16 sequences
of length 64, a replay ratio of 1024, and 1.1 million environment steps. It logs
evaluation return against environment steps to JSON so curves can be compared
without relying on wall-clock-dependent training iterations.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

For a three-seed median and interquartile reproduction run:

```bash
python sota-implementations/dreamer_v3/benchmark.py \
  --seeds 0 1 2 \
  --output-dir dmc_walker_runs
```

The benchmark writes one metrics file per seed plus `summary.json` and checks a
minimum final median return of 700. Use `--minimum-final-return` to override the
acceptance threshold when evaluating a deliberately smaller ablation. Full
learning-curve runs are intended for scheduled or manual validation; pull-request
CI uses short smoke overrides.
