# Flaky Test Report - 2026-05-10

## Summary

- **Flaky tests**: 130
- **Newly flaky** (last 7 days): 130
- **Resolved**: 0
- **Total tests analyzed**: 27451
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...nforce_value_net[False-True-ValueEstimators.TD1-gae-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...force_value_net[False-True-ValueEstimators.TD1-gae-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...inforce_value_net[False-True-ValueEstimators.TD1-td-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...nforce_value_net[False-True-ValueEstimators.TD1-td-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `..._value_net[False-True-ValueEstimators.TD1-td_lambda-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...value_net[False-True-ValueEstimators.TD1-td_lambda-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...force_value_net[False-True-ValueEstimators.TD1-None-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...orce_value_net[False-True-ValueEstimators.TD1-None-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...nforce_value_net[False-True-ValueEstimators.TD0-gae-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...force_value_net[False-True-ValueEstimators.TD0-gae-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...inforce_value_net[False-True-ValueEstimators.TD0-td-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...nforce_value_net[False-True-ValueEstimators.TD0-td-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `..._value_net[False-True-ValueEstimators.TD0-td_lambda-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...value_net[False-True-ValueEstimators.TD0-td_lambda-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...force_value_net[False-True-ValueEstimators.TD0-None-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...orce_value_net[False-True-ValueEstimators.TD0-None-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...nforce_value_net[False-True-ValueEstimators.GAE-gae-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...force_value_net[False-True-ValueEstimators.GAE-gae-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...inforce_value_net[False-True-ValueEstimators.GAE-td-True]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |
| `...nforce_value_net[False-True-ValueEstimators.GAE-td-False]` 🆕 | 20.0% (30/150) | 30 | 0.40 | 2026-05-06 |


### Newly Flaky Tests

- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD1-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TD0-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.GAE-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-ValueEstimators.TDLambda-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-True-None-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD1-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TD0-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.GAE-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-ValueEstimators.TDLambda-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[False-False-None-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD1-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TD0-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.GAE-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-ValueEstimators.TDLambda-None-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-gae-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-gae-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-td-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-td-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-td_lambda-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-td_lambda-False]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-None-True]`
- `test/objectives/test_ppo.py::TestReinforce::test_reinforce_value_net[True-True-None-None-False]`
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

*Generated at 2026-05-10T07:04:07.776933+00:00*