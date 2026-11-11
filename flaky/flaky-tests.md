# Flaky Test Report - 2026-11-11

## Summary

- **Flaky tests**: 2
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32972
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...TestDataConfigs::test_complex_replay_buffer_configuration` 🆕 | 13.8% (22/160) | 22 | 0.28 | 2026-11-08 |
| `...stDreamerV3Components::test_discrete_actor[cuda-autocast]` 🆕 | 5.4% (4/74) | 4 | 0.09 | 2026-11-07 |


### Newly Flaky Tests

- `test/test_configs.py::TestDataConfigs::test_complex_replay_buffer_configuration`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-11T06:20:39.581193+00:00*