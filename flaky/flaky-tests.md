# Flaky Test Report - 2026-10-09

## Summary

- **Flaky tests**: 5772
- **Newly flaky** (last 7 days): 40
- **Resolved**: 0
- **Total tests analyzed**: 32507
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ripts/test_examples.py::test_example[distributed-ray-dqn]` 🆕 | 50.0% (10/20) | 10 | 1.00 | 2026-10-09 |
| `...scripts/test_examples.py::test_example[ray-wandb-monitor]` 🆕 | 50.0% (10/20) | 10 | 1.00 | 2026-10-09 |
| `...ts/test_examples.py::test_example[services-ray-collector]` 🆕 | 50.0% (10/20) | 10 | 1.00 | 2026-10-09 |
| `....py::TestCollectorStats::test_ray_stats_during_collection` 🆕 | 54.3% (89/164) | 89 | 0.91 | 2026-10-09 |
| `...llectors.py::TestCollectorStats::test_ray_collector_stats` 🆕 | 54.3% (89/164) | 89 | 0.91 | 2026-10-09 |
| `...re::test_static_batch_pads_slices_and_owns_results[False]` | 59.5% (44/74) | 44 | 0.81 | 2026-10-07 |
| `...cessSlotTransport::test_server_batched_pass_on_cuda[True]` | 64.9% (48/74) | 48 | 0.70 | 2026-10-08 |
| `...y::TestRayCollector::test_distributed_collector_basic[50]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...::TestRayCollector::test_distributed_collector_basic[100]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ted.py::TestRayCollector::test_distributed_collector_mult` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...tributed.py::TestRayCollector::test_collector_next_method` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ibuted.py::TestRayCollector::test_dqn_trainer_ray_backend` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...ollector::test_offpolicy_trainer_ray_backend[DDPGTrainer]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...Collector::test_offpolicy_trainer_ray_backend[SACTrainer]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...Collector::test_offpolicy_trainer_ray_backend[TD3Trainer]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...tor::test_ray_learner_publishes_to_collector_owned_policy` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...py::TestRayCollector::test_ray_owned_inference_and_replay` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...RayCollector::test_ray_collector_pause_drains_and_resumes` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...:TestRayCollector::test_distributed_collector_sync[False]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |
| `...::TestRayCollector::test_distributed_collector_sync[True]` 🆕 | 66.7% (40/60) | 40 | 0.67 | 2026-10-09 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[distributed-ray-dqn]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[ray-wandb-monitor]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[services-ray-collector]`
- `test/test_collectors.py::TestCollectorStats::test_ray_stats_during_collection`
- `test/test_collectors.py::TestCollectorStats::test_ray_collector_stats`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_basic[50]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_basic[100]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_mult`
- `test/test_distributed.py::TestRayCollector::test_collector_next_method`
- `test/test_distributed.py::TestRayCollector::test_dqn_trainer_ray_backend`
- `test/test_distributed.py::TestRayCollector::test_offpolicy_trainer_ray_backend[DDPGTrainer]`
- `test/test_distributed.py::TestRayCollector::test_offpolicy_trainer_ray_backend[SACTrainer]`
- `test/test_distributed.py::TestRayCollector::test_offpolicy_trainer_ray_backend[TD3Trainer]`
- `test/test_distributed.py::TestRayCollector::test_ray_learner_publishes_to_collector_owned_policy`
- `test/test_distributed.py::TestRayCollector::test_ray_owned_inference_and_replay`
- `test/test_distributed.py::TestRayCollector::test_ray_collector_pause_drains_and_resumes`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_sync[False]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_sync[True]`
- `test/test_distributed.py::TestRayCollector::test_collector_shutdown_clears_python_processes[False]`
- `test/test_distributed.py::TestRayCollector::test_collector_shutdown_clears_python_processes[True]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_class[MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_class[MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_class[Collector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-False-Collector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-False-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-False-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-True-Collector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-True-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[False-True-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-False-Collector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-False-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-False-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-True-Collector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-True-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-True-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_ray_collector_policy_constructor`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape0-ns0]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape0-ns1]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape1-ns0]`
- `test/test_specs.py::TestRanges::test_multi_discrete_conversion_batched_spec[device0-shape1-ns1]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-09T06:32:13.413694+00:00*