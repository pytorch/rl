# DreamerV3UpdateRatio

*class*torchrl.trainers.algorithms.DreamerV3UpdateRatio(*ratio: float*)[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3UpdateRatio)

Schedule learner updates from a ratio of updates to driver records.

Each call truncates the count from the cumulative driver-record count and
keeps the remainder. The first call returns one update.

Parameters:

**ratio** (*float*) - Learner updates for each driver record. Non-positive values
disable updates.

Examples

```
>>> from torchrl.trainers.algorithms import DreamerV3UpdateRatio
>>> schedule = DreamerV3UpdateRatio(0.25)
>>> schedule(4), schedule(6)
(1, 0)
>>> saved = schedule.state_dict()
>>> expected = schedule(8)
>>> schedule.load_state_dict(saved)
>>> schedule(8) == expected
True
```

See also

[`DreamerV3UpdateRatioConfig`](torchrl.trainers.algorithms.configs.DreamerV3UpdateRatioConfig.html#torchrl.trainers.algorithms.configs.DreamerV3UpdateRatioConfig)

load_state_dict(*state_dict: Mapping[str, float | None]*) → None[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3UpdateRatio.load_state_dict)

Restore the update schedule's progress and ratio.

reset(*record_count: int*) → None[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3UpdateRatio.reset)

Discard owed updates and start counting after `record_count` records.

Use when rebuilding replay after a resume without saved replay, so
collection warm-up does not accumulate a catch-up update burst.

state_dict() → dict[str, float | None][[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3UpdateRatio.state_dict)

Return the ratio and cumulative progress, including fractional updates.