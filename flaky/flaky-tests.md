# Flaky Test Report - 2026-10-06

## Summary

- **Flaky tests**: 5768
- **Newly flaky** (last 7 days): 36
- **Resolved**: 0
- **Total tests analyzed**: 32498
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...y::TestRayCollector::test_distributed_collector_basic[50]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...::TestRayCollector::test_distributed_collector_basic[100]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...ted.py::TestRayCollector::test_distributed_collector_mult` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...tributed.py::TestRayCollector::test_collector_next_method` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...ibuted.py::TestRayCollector::test_dqn_trainer_ray_backend` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...ollector::test_offpolicy_trainer_ray_backend[DDPGTrainer]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...Collector::test_offpolicy_trainer_ray_backend[SACTrainer]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...Collector::test_offpolicy_trainer_ray_backend[TD3Trainer]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...tor::test_ray_learner_publishes_to_collector_owned_policy` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...py::TestRayCollector::test_ray_owned_inference_and_replay` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...RayCollector::test_ray_collector_pause_drains_and_resumes` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...:TestRayCollector::test_distributed_collector_sync[False]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...::TestRayCollector::test_distributed_collector_sync[True]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...r::test_collector_shutdown_clears_python_processes[False]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...or::test_collector_shutdown_clears_python_processes[True]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...tor::test_distributed_collector_class[MultiSyncCollector]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...or::test_distributed_collector_class[MultiAsyncCollector]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...RayCollector::test_distributed_collector_class[Collector]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...distributed_collector_updatepolicy[False-False-Collector]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |
| `...ed_collector_updatepolicy[False-False-MultiSyncCollector]` 🆕 | 21.4% (12/56) | 12 | 0.43 | 2026-10-05 |


### Newly Flaky Tests

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
- `test/test_collectors.py::TestCollectorStats::test_ray_stats_during_collection`
- `test/test_collectors.py::TestCollectorStats::test_ray_collector_stats`
- `.github/unittest/examples/scripts/test_examples.py::test_example[services-ray-collector]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[distributed-ray-dqn]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[ray-wandb-monitor]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-06T06:27:53.940192+00:00*