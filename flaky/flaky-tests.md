# Flaky Test Report - 2026-09-22

## Summary

- **Flaky tests**: 72
- **Newly flaky** (last 7 days): 0
- **Resolved**: 0
- **Total tests analyzed**: 31834
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...mposite::test_device_cast[dest1-shape0-dtype0-None-False]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...osite::test_device_cast[dest1-shape0-dtype0-device1-True]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...site::test_device_cast[dest1-shape0-dtype0-device1-False]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...omposite::test_device_cast[dest1-shape0-dtype1-None-True]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...mposite::test_device_cast[dest1-shape0-dtype1-None-False]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...osite::test_device_cast[dest1-shape0-dtype1-device1-True]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...site::test_device_cast[dest1-shape0-dtype1-device1-False]` | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-09-22T06:30:00.213953+00:00*