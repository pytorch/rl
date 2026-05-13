# Flaky Test Report - 2026-05-13

## Summary

- **Flaky tests**: 10
- **Newly flaky** (last 7 days): 10
- **Resolved**: 0
- **Total tests analyzed**: 27522
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ted_collector_updatepolicy[True-False-MultiSyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...ed_collector_updatepolicy[True-False-MultiAsyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...uted_collector_updatepolicy[True-True-MultiSyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...ted_collector_updatepolicy[True-True-MultiAsyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...ted_collector_updatepolicy[True-False-MultiSyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...ed_collector_updatepolicy[True-False-MultiAsyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...uted_collector_updatepolicy[True-True-MultiSyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...ted_collector_updatepolicy[True-True-MultiAsyncCollector]` 🆕 | 15.4% (8/52) | 8 | 0.31 | 2026-05-08 |
| `...tory::test_per_worker_weight_sync_with_distinct_factories` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-05-08 |
| `...tory::test_per_worker_weight_sync_multiple_workers_update` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-05-08 |


### Newly Flaky Tests

- `test/test_distributed.py::TestRPCCollector::test_distributed_collector_updatepolicy[True-False-MultiSyncCollector]`
- `test/test_distributed.py::TestRPCCollector::test_distributed_collector_updatepolicy[True-False-MultiAsyncCollector]`
- `test/test_distributed.py::TestRPCCollector::test_distributed_collector_updatepolicy[True-True-MultiSyncCollector]`
- `test/test_distributed.py::TestRPCCollector::test_distributed_collector_updatepolicy[True-True-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-False-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-False-MultiAsyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-True-MultiSyncCollector]`
- `test/test_distributed.py::TestRayCollector::test_distributed_collector_updatepolicy[True-True-MultiAsyncCollector]`
- `test/test_collectors.py::TestMakePolicyFactory::test_per_worker_weight_sync_with_distinct_factories`
- `test/test_collectors.py::TestMakePolicyFactory::test_per_worker_weight_sync_multiple_workers_update`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-05-13T07:10:58.661605+00:00*