# Flaky Test Report - 2026-06-13

## Summary

- **Flaky tests**: 2
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 29088
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...::test_multi_sync_data_collector_ordering[stack-True-8-5]` 🆕 | 13.5% (20/148) | 20 | 0.27 | 2026-06-12 |
| `..._configs.py::TestDataConfigs::test_writer_ensemble_config` 🆕 | 6.8% (10/148) | 10 | 0.14 | 2026-06-12 |


### Newly Flaky Tests

- `test/test_collectors.py::TestCollectorGeneric::test_multi_sync_data_collector_ordering[stack-True-8-5]`
- `test/test_configs.py::TestDataConfigs::test_writer_ensemble_config`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-06-13T07:26:01.529402+00:00*