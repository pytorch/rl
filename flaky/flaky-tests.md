# Flaky Test Report - 2026-11-17

## Summary

- **Flaky tests**: 3
- **Newly flaky** (last 7 days): 1
- **Resolved**: 0
- **Total tests analyzed**: 32982
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...TestDataConfigs::test_complex_replay_buffer_configuration` | 15.1% (24/159) | 24 | 0.30 | 2026-11-12 |
| `...scripts/test_sota.py::test_commands[discrete_cql_offline]` 🆕 | 11.1% (3/27) | 3 | 0.13 | 2026-11-17 |
| `...stDreamerV3Components::test_discrete_actor[cuda-autocast]` | 5.5% (4/73) | 4 | 0.09 | 2026-11-07 |


### Newly Flaky Tests

- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[discrete_cql_offline]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-17T06:19:45.925313+00:00*