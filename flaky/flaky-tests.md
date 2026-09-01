# Flaky Test Report - 2026-08-31

## Summary

- **Flaky tests**: 2
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32428
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `..._block_gru_triton_gradient_parity[SiLU-4-1-2-3-dtype4-48]` 🆕 | 39.3% (22/56) | 22 | 0.79 | 2026-08-30 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-08-28 |


### Newly Flaky Tests

- `test/modules/test_dreamer_components.py::test_public_block_gru_triton_gradient_parity[SiLU-4-1-2-3-dtype4-48]`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-31T06:35:06.428682+00:00*