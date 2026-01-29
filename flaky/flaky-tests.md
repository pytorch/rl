# Flaky Test Report - 2026-01-29

## Summary

- **Flaky tests**: 4
- **Newly flaky** (last 7 days): 4
- **Resolved**: 0
- **Total tests analyzed**: 26744
- **CI runs analyzed**: 150

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]` 🆕 | 10.0% (2/20) | 2 | 0.08 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]` 🆕 | 10.0% (2/20) | 2 | 0.08 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]` 🆕 | 10.0% (2/20) | 2 | 0.08 | 2026-01-29 |
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` 🆕 | 10.0% (2/20) | 2 | 0.08 | 2026-01-29 |


### Newly Flaky Tests

- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]`
- `test/test_rb.py::TestLazyMemmapStorageCleanup::test_cleanup_registry`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-01-29T13:37:49.752658+00:00*