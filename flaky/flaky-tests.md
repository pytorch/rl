# Flaky Test Report - 2026-11-02

## Summary

- **Flaky tests**: 7
- **Newly flaky** (last 7 days): 5
- **Resolved**: 0
- **Total tests analyzed**: 32893
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `.../test_examples.py::test_example[replay-buffer-checkpoint]` 🆕 | 40.0% (8/20) | 8 | 0.80 | 2026-11-02 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-11-01 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-11-01 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-11-01 |
| `...stDreamerV3Components::test_discrete_actor[cuda-autocast]` | 8.0% (6/75) | 6 | 0.16 | 2026-10-31 |
| `....py::TestHydraParsing::test_dqn_trainer_parsing_with_file` | 6.1% (10/165) | 10 | 0.12 | 2026-10-29 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-10-30 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-02T06:26:23.999755+00:00*