# Flaky Test Report - 2026-08-16

## Summary

- **Flaky tests**: 10
- **Newly flaky** (last 7 days): 10
- **Resolved**: 0
- **Total tests analyzed**: 31526
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...torages.py::TestStorages::test__rand_given_ndim_recompile` 🆕 | 9.8% (16/163) | 16 | 0.20 | 2026-08-12 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 8.0% (13/163) | 13 | 0.16 | 2026-08-15 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 8.0% (13/163) | 13 | 0.16 | 2026-08-15 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-08-15 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-08-15 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-08-15 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 7.7% (10/130) | 10 | 0.15 | 2026-08-15 |
| `...core.py::test_replay_buffer_prefetch_state_dict_roundtrip` 🆕 | 6.1% (2/33) | 2 | 0.05 | 2026-08-15 |
| `...replay_buffer_load_state_dict_waits_for_in_flight_samples` 🆕 | 6.1% (2/33) | 2 | 0.05 | 2026-08-15 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` 🆕 | 6.1% (2/33) | 2 | 0.05 | 2026-08-15 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorages::test__rand_given_ndim_recompile`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_state_dict_roundtrip`
- `test/rb/test_rb_core.py::test_replay_buffer_load_state_dict_waits_for_in_flight_samples`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-16T06:14:21.783597+00:00*