# Flaky Test Report - 2026-11-04

## Summary

- **Flaky tests**: 5
- **Newly flaky** (last 7 days): 5
- **Resolved**: 0
- **Total tests analyzed**: 32896
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `.../test_examples.py::test_example[replay-buffer-checkpoint]` 🆕 | 55.6% (10/18) | 10 | 0.89 | 2026-11-02 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 8.4% (11/131) | 11 | 0.17 | 2026-11-03 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 8.4% (11/131) | 11 | 0.17 | 2026-11-03 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 8.4% (11/131) | 11 | 0.17 | 2026-11-03 |
| `...TestDataConfigs::test_complex_replay_buffer_configuration` 🆕 | 7.3% (12/164) | 12 | 0.15 | 2026-11-03 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
- `test/test_configs.py::TestDataConfigs::test_complex_replay_buffer_configuration`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-04T06:23:01.456795+00:00*