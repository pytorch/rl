# Flaky Test Report - 2026-08-13

## Summary

- **Flaky tests**: 3
- **Newly flaky** (last 7 days): 3
- **Resolved**: 0
- **Total tests analyzed**: 31382
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...torages.py::TestStorages::test__rand_given_ndim_recompile` 🆕 | 24.4% (40/164) | 40 | 0.49 | 2026-08-12 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-08-12 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` 🆕 | 5.5% (9/164) | 9 | 0.11 | 2026-08-12 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorages::test__rand_given_ndim_recompile`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_block_gru_torch_compile`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[TensorDictReplayBufferConfig]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-13T06:39:30.747584+00:00*