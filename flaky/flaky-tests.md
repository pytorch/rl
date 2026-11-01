# Flaky Test Report - 2026-11-01

## Summary

- **Flaky tests**: 51
- **Newly flaky** (last 7 days): 7
- **Resolved**: 0
- **Total tests analyzed**: 32875
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `.../test_examples.py::test_example[replay-buffer-checkpoint]` 🆕 | 30.4% (7/23) | 7 | 0.61 | 2026-11-01 |
| `...stDreamerV3Components::test_discrete_actor[cuda-autocast]` 🆕 | 8.0% (6/75) | 6 | 0.16 | 2026-10-31 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-11-01 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-11-01 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-11-01 |
| `....py::TestHydraParsing::test_dqn_trainer_parsing_with_file` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-10-29 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-10-30 |
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` | 13.3% (2/15) | 2 | 0.11 | 2026-10-24 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-01T06:15:45.243658+00:00*