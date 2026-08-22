# Flaky Test Report - 2026-08-22

## Summary

- **Flaky tests**: 10
- **Newly flaky** (last 7 days): 10
- **Resolved**: 0
- **Total tests analyzed**: 31570
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...stDiffusionActor::test_reduced_precision_schedule[dtype0]` 🆕 | 9.4% (10/106) | 10 | 0.19 | 2026-08-21 |
| `...s/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]` 🆕 | 9.4% (12/128) | 12 | 0.19 | 2026-08-21 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 9.3% (13/139) | 13 | 0.19 | 2026-08-21 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` 🆕 | 9.3% (13/139) | 13 | 0.19 | 2026-08-21 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` 🆕 | 9.3% (13/139) | 13 | 0.19 | 2026-08-21 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` 🆕 | 9.3% (13/139) | 13 | 0.19 | 2026-08-21 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` 🆕 | 9.3% (13/139) | 13 | 0.19 | 2026-08-21 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 7.9% (11/139) | 11 | 0.16 | 2026-08-20 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` 🆕 | 7.2% (10/139) | 10 | 0.14 | 2026-08-21 |
| `...core.py::test_replay_buffer_prefetch_state_dict_roundtrip` 🆕 | 5.8% (8/139) | 8 | 0.12 | 2026-08-21 |


### Newly Flaky Tests

- `test/modules/test_actor.py::TestDiffusionActor::test_reduced_precision_schedule[dtype0]`
- `test/objectives/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]`
- `test/test_distributions.py::TestTanhNormal::test_tanhnormal_rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip`
- `test/rb/test_rb_core.py::test_replay_buffer_prefetch_state_dict_roundtrip`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-22T06:11:46.911497+00:00*