# Flaky Test Report - 2026-08-14

## Summary

- **Flaky tests**: 7
- **Newly flaky** (last 7 days): 7
- **Resolved**: 0
- **Total tests analyzed**: 31513
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...torages.py::TestStorages::test__rand_given_ndim_recompile` 🆕 | 24.5% (40/163) | 40 | 0.49 | 2026-08-12 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 8.0% (13/163) | 13 | 0.16 | 2026-08-13 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 7.4% (8/108) | 8 | 0.15 | 2026-08-13 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 6.2% (4/64) | 4 | 0.10 | 2026-08-13 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 6.2% (4/64) | 4 | 0.10 | 2026-08-13 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 6.2% (4/64) | 4 | 0.10 | 2026-08-13 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 6.2% (4/64) | 4 | 0.10 | 2026-08-13 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorages::test__rand_given_ndim_recompile`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-14T06:37:59.010370+00:00*