# Flaky Test Report - 2026-10-11

## Summary

- **Flaky tests**: 5773
- **Newly flaky** (last 7 days): 5
- **Resolved**: 0
- **Total tests analyzed**: 32507
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `....py::TestCollectorStats::test_ray_stats_during_collection` | 54.6% (89/163) | 89 | 0.91 | 2026-10-09 |
| `...llectors.py::TestCollectorStats::test_ray_collector_stats` | 54.6% (89/163) | 89 | 0.91 | 2026-10-09 |
| `...cessSlotTransport::test_server_batched_pass_on_cuda[True]` | 45.1% (32/71) | 32 | 0.90 | 2026-10-08 |
| `...ts/test_examples.py::test_example[services-ray-collector]` | 55.6% (10/18) | 10 | 0.89 | 2026-10-09 |
| `...ripts/test_examples.py::test_example[distributed-ray-dqn]` | 55.6% (10/18) | 10 | 0.89 | 2026-10-09 |
| `...scripts/test_examples.py::test_example[ray-wandb-monitor]` | 55.6% (10/18) | 10 | 0.89 | 2026-10-09 |
| `...re::test_static_batch_pads_slices_and_owns_results[False]` | 39.4% (28/71) | 28 | 0.79 | 2026-10-07 |
| `...y::TestRayCollector::test_distributed_collector_basic[50]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...::TestRayCollector::test_distributed_collector_basic[100]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ted.py::TestRayCollector::test_distributed_collector_mult` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...tributed.py::TestRayCollector::test_collector_next_method` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ibuted.py::TestRayCollector::test_dqn_trainer_ray_backend` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ollector::test_offpolicy_trainer_ray_backend[DDPGTrainer]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...Collector::test_offpolicy_trainer_ray_backend[SACTrainer]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...Collector::test_offpolicy_trainer_ray_backend[TD3Trainer]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...tor::test_ray_learner_publishes_to_collector_owned_policy` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...py::TestRayCollector::test_ray_owned_inference_and_replay` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...RayCollector::test_ray_collector_pause_drains_and_resumes` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...:TestRayCollector::test_distributed_collector_sync[False]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...::TestRayCollector::test_distributed_collector_sync[True]` | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |


### Newly Flaky Tests

- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape0-ns0]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape0-ns1]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape1-ns0]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape1-ns1]`
- `.github/unittest/examples/scripts/test_examples.py::test_run_example_terminates_timeouts`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-11T06:24:55.945841+00:00*