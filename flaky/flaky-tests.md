# Flaky Test Report - 2026-08-19

## Summary

- **Flaky tests**: 9
- **Newly flaky** (last 7 days): 9
- **Resolved**: 0
- **Total tests analyzed**: 31554
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...stDiffusionActor::test_reduced_precision_schedule[dtype0]` 🆕 | 9.6% (7/73) | 7 | 0.19 | 2026-08-18 |
| `...s/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]` 🆕 | 9.5% (9/95) | 9 | 0.19 | 2026-08-18 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 8.7% (12/138) | 12 | 0.17 | 2026-08-18 |
| `...core.py::test_replay_buffer_prefetch_state_dict_roundtrip` 🆕 | 5.8% (8/138) | 8 | 0.12 | 2026-08-18 |


### Newly Flaky Tests

- `test/modules/test_actor.py::TestDiffusionActor::test_reduced_precision_schedule[dtype0]`
- `test/objectives/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_state_dict_roundtrip`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-19T06:15:50.220035+00:00*