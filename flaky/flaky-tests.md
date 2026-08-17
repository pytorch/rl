# Flaky Test Report - 2026-08-17

## Summary

- **Flaky tests**: 9
- **Newly flaky** (last 7 days): 9
- **Resolved**: 0
- **Total tests analyzed**: 31536
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-08-16 |
| `...core.py::test_replay_buffer_prefetch_state_dict_roundtrip` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-08-16 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` 🆕 | 5.3% (4/76) | 4 | 0.08 | 2026-08-16 |
| `...s/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]` 🆕 | 9.1% (2/22) | 2 | 0.07 | 2026-08-16 |


### Newly Flaky Tests

- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_state_dict_roundtrip`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip`
- `test/objectives/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-17T06:21:51.300255+00:00*