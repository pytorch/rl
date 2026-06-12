# Flaky Test Report - 2026-06-12

## Summary

- **Flaky tests**: 2
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 29083
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...::test_multi_sync_data_collector_ordering[stack-True-8-5]` 🆕 | 12.2% (18/147) | 18 | 0.24 | 2026-06-11 |
| `..._configs.py::TestDataConfigs::test_writer_ensemble_config` 🆕 | 5.4% (8/147) | 8 | 0.11 | 2026-06-11 |


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

*Generated at 2026-06-12T07:47:26.830693+00:00*