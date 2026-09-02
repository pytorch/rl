# Flaky Test Report - 2026-09-02

## Summary

- **Flaky tests**: 2
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32442
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `..._block_gru_triton_gradient_parity[SiLU-4-1-2-3-dtype4-48]` 🆕 | 30.0% (18/60) | 18 | 0.60 | 2026-08-30 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-08-28 |


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

*Generated at 2026-09-02T06:35:03.665072+00:00*