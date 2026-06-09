# Flaky Test Report - 2026-06-09

## Summary

- **Flaky tests**: 1
- **Newly flaky** (last 7 days): 1
- **Resolved**: 0
- **Total tests analyzed**: 28662
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...stLSTMModule::test_scan_backend_backward_matches_pad[gru]` 🆕 | 6.7% (10/149) | 10 | 0.13 | 2026-06-04 |


### Newly Flaky Tests

- `test/modules/test_rnn.py::TestLSTMModule::test_scan_backend_backward_matches_pad[gru]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-06-09T07:22:36.990213+00:00*