# Flaky Test Report - 2026-10-31

## Summary

- **Flaky tests**: 78
- **Newly flaky** (last 7 days): 4
- **Resolved**: 0
- **Total tests analyzed**: 32838
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype0-None-False]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype0-device1-True]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype0-device1-False]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype1-None-True]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype1-None-False]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype1-device1-True]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype1-device1-False]` | 33.3% (5/15) | 5 | 0.67 | 2026-10-24 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-31T06:17:33.632959+00:00*