# Flaky Test Report - 2026-01-29

## Summary

- **Flaky tests**: 5
- **Newly flaky** (last 7 days): 5
- **Resolved**: 0
- **Total tests analyzed**: 26758
- **CI runs analyzed**: 150

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...b.py::TestLazyMemmapStorageCleanup::test_cleanup_registry` 🆕 | 16.0% (8/50) | 8 | 0.32 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]` 🆕 | 8.0% (4/50) | 4 | 0.13 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]` 🆕 | 8.0% (4/50) | 4 | 0.13 | 2026-01-29 |
| `...nvs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]` 🆕 | 8.0% (4/50) | 4 | 0.13 | 2026-01-29 |
| `...t_objectives.py::TestSAC::test_sac_prioritized_weights[2]` 🆕 | 8.0% (4/50) | 4 | 0.13 | 2026-01-29 |


### Newly Flaky Tests

- `test/test_rb.py::TestLazyMemmapStorageCleanup::test_cleanup_registry`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size0-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size1-0]`
- `test/test_envs.py::TestMultiKeyEnvs::test_rollout[2-5-batch_size2-0]`
- `test/test_objectives.py::TestSAC::test_sac_prioritized_weights[2]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-01-29T17:57:19.069613+00:00*