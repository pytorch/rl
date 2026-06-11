# Flaky Test Report - 2026-06-11

## Summary

- **Flaky tests**: 3
- **Newly flaky** (last 7 days): 3
- **Resolved**: 0
- **Total tests analyzed**: 28908
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...est_custom_envs.py::TestPendulum::test_pendulum_env[None]` 🆕 | 6.7% (10/149) | 10 | 0.13 | 2026-06-10 |
| `..._custom_envs.py::TestPendulum::test_pendulum_env[device1]` 🆕 | 6.7% (10/149) | 10 | 0.13 | 2026-06-10 |
| `...::test_multi_sync_data_collector_ordering[stack-True-8-5]` 🆕 | 5.4% (8/149) | 8 | 0.11 | 2026-06-10 |


### Newly Flaky Tests

- `test/test_custom_envs.py::TestPendulum::test_pendulum_env[None]`
- `test/test_custom_envs.py::TestPendulum::test_pendulum_env[device1]`
- `test/test_collectors.py::TestCollectorGeneric::test_multi_sync_data_collector_ordering[stack-True-8-5]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-06-11T07:51:08.354730+00:00*