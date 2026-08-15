# Flaky Test Report - 2026-08-15

## Summary

- **Flaky tests**: 7
- **Newly flaky** (last 7 days): 7
- **Resolved**: 0
- **Total tests analyzed**: 31526
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...torages.py::TestStorages::test__rand_given_ndim_recompile` 🆕 | 19.6% (32/163) | 32 | 0.39 | 2026-08-12 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 8.0% (13/163) | 13 | 0.16 | 2026-08-14 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 7.9% (12/152) | 12 | 0.16 | 2026-08-14 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 7.4% (8/108) | 8 | 0.15 | 2026-08-14 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 7.4% (8/108) | 8 | 0.15 | 2026-08-14 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 7.4% (8/108) | 8 | 0.15 | 2026-08-14 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 7.4% (8/108) | 8 | 0.15 | 2026-08-14 |


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

*Generated at 2026-08-15T06:14:42.942296+00:00*