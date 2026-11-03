# Flaky Test Report - 2026-11-03

## Summary

- **Flaky tests**: 7
- **Newly flaky** (last 7 days): 7
- **Resolved**: 0
- **Total tests analyzed**: 32896
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `.../test_examples.py::test_example[replay-buffer-checkpoint]` 🆕 | 55.6% (10/18) | 10 | 0.89 | 2026-11-02 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 8.3% (10/120) | 10 | 0.17 | 2026-11-03 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 8.3% (10/120) | 10 | 0.17 | 2026-11-03 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 8.3% (10/120) | 10 | 0.17 | 2026-11-03 |
| `...TestDataConfigs::test_complex_replay_buffer_configuration` 🆕 | 7.3% (12/164) | 12 | 0.15 | 2026-11-03 |
| `...py::TestHydraParsing::test_ddpg_trainer_parsing_with_file` 🆕 | 6.1% (10/164) | 10 | 0.12 | 2026-11-03 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 6.1% (10/164) | 10 | 0.12 | 2026-10-30 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
- `test/test_configs.py::TestDataConfigs::test_complex_replay_buffer_configuration`
- `test/test_configs.py::TestHydraParsing::test_ddpg_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-03T06:21:14.826882+00:00*