# Flaky Test Report - 2026-10-26

## Summary

- **Flaky tests**: 5780
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32765
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype0-None-False]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype0-device1-True]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype0-device1-False]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype1-None-True]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype1-None-False]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype1-device1-True]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype1-device1-False]` | 80.0% (12/15) | 12 | 0.40 | 2026-10-24 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorageExport::test_device_storage[cuda-ListStorage-True]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-26T06:31:15.304577+00:00*